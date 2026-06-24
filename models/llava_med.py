import sys
import os
import torch
import torch.nn as nn
from typing import Optional, Tuple
import copy

LLAVA_MED_REPO = "/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/projects/LLaVA-Med"
LLAVA_MED_WEIGHTS = "/media/tai/002dda08-6217-423d-9b45-31f72c49d1c5/ilab/Ross-rao/downloads/llava-med-v1.5-mistral-7b"

if LLAVA_MED_REPO not in sys.path:
    sys.path.insert(0, LLAVA_MED_REPO)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle


class LlavaMed:

    def __init__(self, device):
        self.device = device
        self.layer = 22  # default steering layer

        model_name = get_model_name_from_path(LLAVA_MED_WEIGHTS)
        self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(
            LLAVA_MED_WEIGHTS,
            model_base=None,
            model_name=model_name,
            device=device,
        )

        self.register = {}
        self.attach_methods = {
            'post_mlp_residual': self._attach_post_mlp_residual,
        }

        self.base_CLIPEncoderLayerPostMlpResidual = copy.deepcopy(
            self._clip_layers[self.layer]
        )

    @property
    def _clip_layers(self):
        return self.model.get_vision_tower().vision_tower.vision_model.encoder.layers

    def encode(self, inputs):
        for hook in self.register.keys():
            self.register[hook] = []
        pixel_values = inputs['pixel_values'].to(self.device, dtype=torch.float16)
        with torch.no_grad():
            output = self.model.get_vision_tower().vision_tower(pixel_values, output_hidden_states=True)
        return output.pooler_output

    def attach(self, attachment_point, layer, sae=None, mean_pool=False):
        if attachment_point in self.attach_methods:
            self.attach_methods[attachment_point](layer, sae, mean_pool=mean_pool)
            self.register[f'{attachment_point}_{layer}'] = []
        else:
            raise NotImplementedError(f"Attachment point {attachment_point} not implemented")

    def _attach_post_mlp_residual(self, layer, sae, mean_pool=False):
        self._clip_layers[layer] = CLIPEncoderLayerPostMlpResidual(
            self._clip_layers[layer],
            sae,
            layer,
            self.register,
            mean_pool=mean_pool,
        )

    def prompt(self, text, image, max_tokens=1024):
        qs = DEFAULT_IMAGE_TOKEN + '\n' + text
        conv = conv_templates["mistral_instruct"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(self.device)

        image_tensor = process_images([image], self.processor, self.model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).half().to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = (
            [KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)]
            if stop_str else []
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=max_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # Strip instruction prefix from Mistral output
        outputs = [x.split('[/INST]')[-1].strip() for x in outputs]
        return outputs

    def attach_and_fix(self, sae, neurons_to_fix={}, pre_zero=False):
        modified_sae = SAEWrapper(sae, neurons_to_fix, pre_zero)
        self._clip_layers[self.layer] = CLIPEncoderLayerPostMlpResidualSteering(
            self.base_CLIPEncoderLayerPostMlpResidual,
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
        x = self.sae.decode(x)
        return x.to(dtype=torch.float16)


class CLIPEncoderLayerPostMlpResidual(nn.Module):
    """Hooks into CLIP encoder layer to capture/modify post-MLP residual activations."""

    def __init__(self, base, sae, layer, register, mean_pool=False):
        super().__init__()
        self.embed_dim = base.embed_dim
        self.self_attn = base.self_attn
        self.layer_norm1 = base.layer_norm1
        self.mlp = base.mlp
        self.layer_norm2 = base.layer_norm2
        self.sae = sae
        self.layer = layer
        self.register = register
        self.mean_pool = mean_pool

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if self.sae is not None:
            input_dtype = hidden_states.dtype
            encoded = self.sae.encode(hidden_states.float())
            stored = encoded.mean(dim=1) if self.mean_pool else encoded
            self.register[f'post_mlp_residual_{self.layer}'].append(stored.detach().cpu())
            hidden_states = self.sae.decode(encoded).to(input_dtype)
        else:
            self.register[f'post_mlp_residual_{self.layer}'].append(hidden_states.detach().cpu())

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class CLIPEncoderLayerPostMlpResidualSteering(nn.Module):
    """Same as above but without register, for steering (neuron clamping)."""

    def __init__(self, base, sae):
        super().__init__()
        self.embed_dim = base.embed_dim
        self.self_attn = base.self_attn
        self.layer_norm1 = base.layer_norm1
        self.mlp = base.mlp
        self.layer_norm2 = base.layer_norm2
        self.sae = sae

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
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
