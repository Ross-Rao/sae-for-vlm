import sys
import copy
import torch
import torch.nn as nn
from typing import Optional, Tuple
from einops import repeat

MED_FLAMINGO_SRC = "/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/projects/Med-Flamingo/src"
MED_FLAMINGO_CHECKPOINT = "/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/downloads/med-flamingo/model.pt"
LLAMA_PATH = "/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/downloads/llama-7b-hf"

if MED_FLAMINGO_SRC not in sys.path:
    sys.path.insert(0, MED_FLAMINGO_SRC)

from open_flamingo import create_model_and_transforms

# open_clip LayerNormFp32 keeps weights in fp32 but its forward() forgets to cast
# fp16 inputs to fp32 before F.layer_norm, causing a dtype mismatch at runtime.
try:
    import open_clip.transformer as _oct
    import torch.nn.functional as _F_patch

    def _ln_fp32_forward_fixed(self, x):
        orig = x.dtype
        x = _F_patch.layer_norm(
            x.float(), self.normalized_shape,
            self.weight.float(),
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return x.to(orig)

    if hasattr(_oct, 'LayerNormFp32'):
        _oct.LayerNormFp32.forward = _ln_fp32_forward_fixed
except Exception:
    pass
import importlib.util as _ilu, os as _os
_spec = _ilu.spec_from_file_location(
    "med_flamingo_src_utils",
    _os.path.join(MED_FLAMINGO_SRC, "utils.py"),
)
_flamingo_utils_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_flamingo_utils_mod)
FlamingoProcessor = _flamingo_utils_mod.FlamingoProcessor


def _clean_generation(response):
    if "Answer:" in response:
        response = response.split("Answer:")[-1]
    response = response.split("<|endofchunk|>")[0]
    response = response.split("\n")[0]
    return response.strip()


class _HFProcessorWrapper:
    """Wraps open_clip Compose transform into HuggingFace-compatible processor interface."""

    def __init__(self, transform):
        self._transform = transform

    def __call__(self, images, return_tensors="pt", padding=True, **kwargs):
        tensors = torch.stack([self._transform(img) for img in images])
        return {"pixel_values": tensors}


class MedFlamingo:

    def __init__(self, device):
        self.device = device
        self.layer = 18  # default hook layer (ViT-L/14 has 24 resblocks)

        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=LLAMA_PATH,
            tokenizer_path=LLAMA_PATH,
            cross_attn_every_n_layers=4,
        )
        model.load_state_dict(
            torch.load(MED_FLAMINGO_CHECKPOINT, map_location="cpu"),
            strict=False,
        )
        model = model.half().to(device)
        model.eval()
        self.model = model
        self.flamingo_processor = FlamingoProcessor(tokenizer, image_processor)
        # HuggingFace-compatible processor for SAE framework (save_activations / encode_images)
        self.processor = _HFProcessorWrapper(image_processor)

        self.register = {}
        self.attach_methods = {
            "post_mlp_residual": self._attach_post_mlp_residual,
        }

        self.base_resblock = copy.deepcopy(self._resblocks[self.layer])

    @property
    def _resblocks(self):
        return self.model.vision_encoder.transformer.resblocks

    def encode(self, inputs):
        """Run vision encoder forward pass, triggering any attached hooks."""
        for key in self.register:
            self.register[key] = []
        pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.float16)
        ve = self.model.vision_encoder
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                x = ve._embeds(pixel_values)
                x = ve.transformer(x)
                pooled, _ = ve._pool(x)
                if ve.proj is not None:
                    pooled = pooled @ ve.proj
        return pooled

    def attach(self, attachment_point, layer, sae=None, mean_pool=False):
        self.layer = layer
        self.base_resblock = copy.deepcopy(self._resblocks[layer])
        if attachment_point not in self.attach_methods:
            raise NotImplementedError(f"Attachment point {attachment_point} not implemented")
        self.attach_methods[attachment_point](layer, sae, mean_pool=mean_pool)
        self.register[f"{attachment_point}_{layer}"] = []

    def _attach_post_mlp_residual(self, layer, sae, mean_pool=False):
        self._resblocks[layer] = OpenCLIPResBlockPostMlpResidual(
            self._resblocks[layer], sae, layer, self.register, mean_pool=mean_pool,
        )

    def attach_and_fix(self, sae, neurons_to_fix={}, pre_zero=False):
        modified_sae = SAEWrapper(sae, neurons_to_fix, pre_zero)
        self._resblocks[self.layer] = OpenCLIPResBlockPostMlpResidualSteering(
            self.base_resblock, modified_sae,
        )

    def prompt(self, text, image, max_tokens=1024):
        prompt_str = (
            "You are a helpful medical assistant. "
            "<image>" + text + "\nAnswer:"
        )
        pixels = self.flamingo_processor.preprocess_images([image])
        pixels = repeat(pixels, "N c h w -> b N T c h w", b=1, T=1).to(self.device, dtype=torch.float16)
        tokenized = self.flamingo_processor.encode_text(prompt_str)

        with torch.inference_mode():
            output_ids = self.model.generate(
                vision_x=pixels,
                lang_x=tokenized["input_ids"].to(self.device),
                attention_mask=tokenized["attention_mask"].to(self.device),
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        response = self.flamingo_processor.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        return [_clean_generation(response)]


class SAEWrapper(nn.Module):

    def __init__(self, sae, neurons_to_fix, pre_zero):
        super().__init__()
        self.sae = sae
        self.neurons_to_fix = neurons_to_fix
        self.pre_zero = pre_zero

    def encode(self, x):
        x = self.sae.encode(x)
        if self.pre_zero:
            x = torch.zeros_like(x)
        for neuron_id, value in self.neurons_to_fix.items():
            x[:, :, neuron_id] = value
        return x

    def decode(self, x):
        return self.sae.decode(x).to(dtype=torch.float16)


class OpenCLIPResBlockPostMlpResidual(nn.Module):
    """Wraps an open_clip ResidualAttentionBlock to capture post-MLP residual activations."""

    def __init__(self, base, sae, layer, register, mean_pool=False):
        super().__init__()
        self.base = base
        self.sae = sae
        self.layer = layer
        self.register = register
        self.mean_pool = mean_pool

    def forward(self, q_x, k_x=None, v_x=None, attn_mask=None):
        x = self.base(q_x, k_x=k_x, v_x=v_x, attn_mask=attn_mask)

        if self.sae is not None:
            input_dtype = x.dtype
            x_enc = self.sae.encode(x.float())
            stored = x_enc.mean(dim=1) if self.mean_pool else x_enc
            self.register[f"post_mlp_residual_{self.layer}"].append(stored.detach().cpu())
            x = self.sae.decode(x_enc).to(input_dtype)
        else:
            stored = x.mean(dim=1) if self.mean_pool else x
            self.register[f"post_mlp_residual_{self.layer}"].append(stored.detach().cpu())
        return x


class OpenCLIPResBlockPostMlpResidualSteering(nn.Module):
    """Same as above but without register, for neuron clamping steering."""

    def __init__(self, base, sae):
        super().__init__()
        self.base = base
        self.sae = sae

    def forward(self, q_x, k_x=None, v_x=None, attn_mask=None):
        x = self.base(q_x, k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        encoded = self.sae.encode(x)
        x = self.sae.decode(encoded)
        return x
