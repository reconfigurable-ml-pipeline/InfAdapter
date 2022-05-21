import argparse
import io
from http import HTTPStatus
import json
import PIL.Image
import base64
import tornado.web
import torch
from kserve.handlers.base import HTTPHandler
from torchvision import models, transforms
import kserve


class KserveModel(kserve.Model):
    def __init__(self, name: str, model_variant: str, model_options: dict):
        super().__init__(name)
        self.current_model = model_variant
        print("cuda is available:", torch.cuda.is_available())
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_options = model_options
        self.load()

    def switch_model(self, new_model: str):
        self.current_model = new_model
        self.load()
        return {"model": self.current_model}

    def load(self) -> bool:
        print("Loading model", self.current_model)
        self.model = self.model_options[self.current_model](pretrained=True)
        self.model.eval()
        print(f"Model {self.current_model} is ready.")
        self.ready = True
        return self.ready


class ModelSwitchHandler(HTTPHandler):
    async def post(self, name: str):
        data = json.loads(self.request.body)
        model: KserveModel = self.get_model(name)
        try:
            response = model.switch_model(data["model"])
        except KeyError:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="selected model is not a valid model name."
            )
        self.write(response)


class ModelServer(kserve.ModelServer):
    def create_application(self):
        application = super().create_application()
        application.add_handlers(r".*", [
            (r"/v3/models/([a-zA-Z0-9_-]+):switch-model", ModelSwitchHandler, dict(models=self.registered_models))
        ])
        return application


def preprocess(inp):
    preprocessor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t[:3, ...]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocessor(PIL.Image.open(io.BytesIO(base64.b64decode(inp)))).unsqueeze(0).tolist()


class Classification(KserveModel):
    def __init__(self, name: str, model_variant: str):
        super().__init__(
            name,
            model_variant,
            {
                "resnet18": models.resnet18,
                "resnet34": models.resnet34,
                "resnet50": models.resnet50,
                "resnet101": models.resnet101,
                "resnet152": models.resnet152
            }
        )

    def preprocess(self, request: dict) -> dict:
        print("preprocess input keys", request.keys())
        return {"instances": preprocess(instance["data"]) for instance in request["instances"]}

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

    def postprocess(self, response: dict) -> dict:
        print("postprocess input keys", response.keys())
        sorted_value = response["sorted_value"]
        sorted_idx = response["sorted_idx"]
        return {
            "top_5_categories": sorted_idx[::-1][:5],
            "top_5_scores": sorted_value[::-1][:5],
            "model": self.current_model
        }


parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument(
    "--model_name", help="The name that the predictor model is served under."
)
parser.add_argument("--model_variant", help="specific model name (e.g: resnet18)")
args = parser.parse_args()

if __name__ == "__main__":
    model = Classification(args.model_name, args.model_variant)
    ModelServer().start([model])
