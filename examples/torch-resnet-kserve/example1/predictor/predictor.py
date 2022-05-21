import argparse

import torch
from torchvision import models
import kserve


class Classification(kserve.Model):
    def __init__(self, name):
        super().__init__(name)
        print("cuda is available:", torch.cuda.is_available())
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load()

    def load(self) -> bool:
        print("Try loading resnet18")
        self.model = getattr(models.resnet, "resnet18")(pretrained=True)
        self.model.eval()
        print("Model is ready.")
        self.ready = True
        return self.ready

    def predict(self, request: dict) -> dict:
        inputs = torch.tensor(request["instances"]).to(self.device)
        print("predict input shape", inputs.shape)
        with torch.no_grad():
            out = self.model(inputs).squeeze()
            sorted_value, sorted_idx = out.sort()
        return {
            "sorted_value": sorted_value.numpy().tolist(),
            "sorted_idx": sorted_idx.numpy().tolist()
        }


parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument(
    "--model_name", help="The name that the predictor model is served under."
)
args = parser.parse_args()

if __name__ == "__main__":
    model = Classification(args.model_name)
    kserve.ModelServer().start([model])
