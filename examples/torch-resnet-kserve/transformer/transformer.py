import os
import io
import PIL.Image
import argparse
import base64

from torchvision import transforms
import kserve


def preprocess(inp):
    preprocessor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t[:3, ...]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocessor(PIL.Image.open(io.BytesIO(base64.b64decode(inp)))).unsqueeze(0).tolist()


class Transformer(kserve.Model):
    def __init__(self, name, predictor_host):
        super().__init__(name)
        self.predictor_host = predictor_host

    def preprocess(self, request: dict) -> dict:
        print("transformer preprocess input keys", request.keys())
        return {"instances": preprocess(instance["data"]) for instance in request["instances"]}

    def postprocess(self, response: dict) -> dict:
        print("transformer postprocess input keys", response.keys())
        sorted_value = response["sorted_value"]
        sorted_idx = response["sorted_idx"]
        return {
            "top_5_categories": sorted_idx[::-1][:5],
            "top_5_scores": sorted_value[::-1][:5]
        }


parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument(
    "--predictor_host", help="The URL for the model predict function", required=True
)
parser.add_argument(
    "--protocol", help="The protocol for the predictor", default="v1"
)
parser.add_argument(
    "--model_name", help="The name that the predictor model is served under."
)
args = parser.parse_args()

if __name__ == "__main__":
    model = Transformer(args.model_name, predictor_host=args.predictor_host)
    kserve.ModelServer(workers=1, http_port=args.http_port).start([model])
