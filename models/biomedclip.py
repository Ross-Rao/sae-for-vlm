import os
import torch
import torch.nn.functional as F
import open_clip


class _TensorBatch(dict):
    """dict with .to(device) for compatibility with existing encode() call signature."""
    def to(self, device):
        return _TensorBatch(
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in self.items()}
        )


class BiomedCLIPProcessor:
    """Wraps open_clip preprocess transform to match HF processor call signature."""
    def __init__(self, preprocess):
        self._preprocess = preprocess

    def __call__(self, images, return_tensors="pt", padding=True):
        tensors = torch.stack([self._preprocess(img) for img in images])
        return _TensorBatch({"pixel_values": tensors})


class _TextEncoding:
    """Mimics HF BatchEncoding: .to(device) + **-unpacking via keys()/__getitem__."""
    def __init__(self, tokens):
        self.input_ids = tokens

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        return self

    def keys(self):
        return ["input_ids"]

    def __getitem__(self, key):
        return getattr(self, key)


class BiomedCLIPTokenizerWrapper:
    """open_clip tokenizer with HF-style call signature."""
    def __init__(self, tokenizer):
        self._tok = tokenizer

    def __call__(self, text, return_tensors="pt", padding=True, truncation=True):
        if isinstance(text, str):
            text = [text]
        tokens = self._tok(text)
        return _TextEncoding(tokens)


class _TextEncodeResult:
    def __init__(self, text_embeds):
        self.text_embeds = text_embeds


class BiomedCLIPTextEncoder:
    """Wraps open_clip text encoder to match CLIPTextModelWithProjection interface."""
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def __call__(self, input_ids=None, **kwargs):
        with torch.no_grad():
            features = self.model.encode_text(input_ids.to(self.device))
            features = F.normalize(features, dim=-1)
        return _TextEncodeResult(features)


class BiomedCLIP:
    MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    def __init__(self, device):
        self.device = device
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        model, _, preprocess = open_clip.create_model_and_transforms(self.MODEL_ID)
        self.model = model.to(device).eval()
        self.processor = BiomedCLIPProcessor(preprocess)
        self._tokenizer = open_clip.get_tokenizer(self.MODEL_ID)

    def encode(self, inputs):
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(pixel_values)
            features = F.normalize(features, dim=-1)
        return features
