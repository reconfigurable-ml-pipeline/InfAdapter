import io
import numpy as np
import PIL.Image
import torch
from torchvision import transforms
from torchvision.models import resnet


def preprocess(inp):
    preprocessor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t[:3, ...]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocessor(PIL.Image.open(io.BytesIO(inp))).unsqueeze(0)


class ClassificationModel:
    def __init__(self):
        self.loaded = False

    def load(self):
        self.model = resnet.resnet18(pretrained=True,)
        self.loaded = True

    def predict(self, inp, feature_names):
        if not self.loaded:
            self.load()
        preprocessed = preprocess(inp)
        with torch.no_grad():
            out = self.model(preprocessed).squeeze()
            sorted_value, sorted_idx = out.sort()
        return {
            "top_5_categories": sorted_idx.numpy().tolist()[::-1][:5],
            "top_5_scores": sorted_value.numpy().tolist()[::-1][:5]
        }

    def health_status(self):
        # Generate dummy input
        _buffer = io.BytesIO()
        PIL.Image.fromarray(np.zeros((720, 720, 3), int), mode="RGB").save(_buffer, "png")
        dummy_png_bytes = _buffer.getvalue()
        return self.predict(dummy_png_bytes, None)
