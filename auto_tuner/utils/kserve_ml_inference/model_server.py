import json

from kserve import ModelServer as KserveModelServer
from kserve.handlers.base import HTTPHandler

from auto_tuner.utils.kserve_ml_inference import CURRENT_MODEL_KEY


class ModelSwitchHandler(HTTPHandler):
    async def post(self, name: str):
        data = json.loads(self.request.body)
        model = self.get_model(name)
        response = await model.switch_model(data[CURRENT_MODEL_KEY])
        self.write(response)


class ModelServer(KserveModelServer):
    def create_application(self):
        application = super().create_application()
        application.add_handlers(r".*", [
            (r"/v3/models/([a-zA-Z0-9_-]+):switch-model", ModelSwitchHandler, dict(models=self.registered_models))
        ])
