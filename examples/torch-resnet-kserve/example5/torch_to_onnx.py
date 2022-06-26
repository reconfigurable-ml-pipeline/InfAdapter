from torchvision.models import resnet
import torch
from torch.autograd import Variable


def build_model(version, device="cpu"):
    model = getattr(resnet, f"resnet{version}")(pretrained=True)
    
    model = model.to(device)

    return model


versions = [18, 34, 50, 101, 152]

for v in versions:
    model = build_model(v)
    dummy_input = Variable(torch.randn(1, 3, 224, 224))
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}
    torch.onnx.export(
        model,
        dummy_input,
        f"onnx/resnet{v}.onnx", 
        verbose=True, 
        input_names=input_names, 
        output_names=output_names, 
        dynamic_axes=dynamic_axes
    )
