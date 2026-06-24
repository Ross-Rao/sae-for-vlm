import os
import copy
import tempfile
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import SiglipImageProcessor

CHEXAGENT_WEIGHTS = "/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/downloads/CheXagent-2-3b"
XRAYSIGLIP_WEIGHTS = "/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/downloads/XraySigLIP__vit-l-16-siglip-384__webli"


class CheXAgent:

    def __init__(self, device):
        self.device = device
        self.layer = 18  # default hook layer (SigLIP ViT-L has 24 layers)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            CHEXAGENT_WEIGHTS, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            CHEXAGENT_WEIGHTS,
            device_map={"" : 0},  # force to RTX 3090; GTX 1080 does not support bfloat16 conv
            trust_remote_code=True,
        )
        self.model = self.model.to(torch.bfloat16)
        self.model.eval()
        self.processor = SiglipImageProcessor.from_pretrained(XRAYSIGLIP_WEIGHTS)

        self.register = {}
        self.attach_methods = {
            "post_mlp_residual": self._attach_post_mlp_residual,
        }

        self.base_SiglipEncoderLayer = copy.deepcopy(
            self._siglip_layers[self.layer]
        )

    @property
    def _siglip_layers(self):
        return self.model.model.visual.model.encoder.layers

    def prompt(self, text, image, max_tokens=512):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            image.convert("RGB").save(tmp_path)
        try:
            query = self.tokenizer.from_list_format([
                {"image": tmp_path},
                {"text": text},
            ])
            conv = [
                {"from": "system", "value": "You are a helpful assistant."},
                {"from": "human", "value": query},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                conv, add_generation_prompt=True, return_tensors="pt"
            )
            with torch.inference_mode():
                output = self.model.generate(
                    input_ids.to(self.device),
                    do_sample=False,
                    num_beams=1,
                    temperature=1.0,
                    top_p=1.0,
                    use_cache=True,
                    max_new_tokens=max_tokens,
                )[0]
            response = self.tokenizer.decode(output[input_ids.size(1):-1])
        finally:
            os.unlink(tmp_path)
        return [response]

    def encode(self, inputs):
        """Run vision encoder forward pass with a preprocessed batch, triggering hooks."""
        for key in self.register:
            self.register[key] = []
        pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            output = self.model.model.visual.model(pixel_values=pixel_values)
        return output.pooler_output

    def attach(self, attachment_point, layer, sae=None):
        if attachment_point not in self.attach_methods:
            raise NotImplementedError(f"Attachment point {attachment_point} not implemented")
        self.attach_methods[attachment_point](layer, sae)
        self.register[f"{attachment_point}_{layer}"] = []

    def _attach_post_mlp_residual(self, layer, sae):
        self._siglip_layers[layer] = SiglipEncoderLayerPostMlpResidual(
            self._siglip_layers[layer],
            sae,
            layer,
            self.register,
        )

    def attach_and_fix(self, sae, neurons_to_fix={}, pre_zero=False):
        modified_sae = SAEWrapper(sae, neurons_to_fix, pre_zero)
        self._siglip_layers[self.layer] = SiglipEncoderLayerPostMlpResidualSteering(
            self.base_SiglipEncoderLayer,
            modified_sae,
        )


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
        return self.sae.decode(x).to(dtype=torch.bfloat16)


class SiglipEncoderLayerPostMlpResidual(nn.Module):
    """Hooks into SigLIP encoder layer to capture/modify post-MLP residual activations."""

    def __init__(self, base, sae, layer, register):
        super().__init__()
        self.self_attn = base.self_attn
        self.layer_norm1 = base.layer_norm1
        self.mlp = base.mlp
        self.layer_norm2 = base.layer_norm2
        self.sae = sae
        self.layer = layer
        self.register = register

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if self.sae is not None:
            hidden_states = self.sae.encode(hidden_states)
            self.register[f"post_mlp_residual_{self.layer}"].append(hidden_states.detach().cpu())
            hidden_states = self.sae.decode(hidden_states)
        else:
            self.register[f"post_mlp_residual_{self.layer}"].append(hidden_states.detach().cpu())

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class SiglipEncoderLayerPostMlpResidualSteering(nn.Module):
    """Same as above but without register, for neuron clamping steering."""

    def __init__(self, base, sae):
        super().__init__()
        self.self_attn = base.self_attn
        self.layer_norm1 = base.layer_norm1
        self.mlp = base.mlp
        self.layer_norm2 = base.layer_norm2
        self.sae = sae

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        encoded = self.sae.encode(hidden_states)
        hidden_states = self.sae.decode(encoded)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs
