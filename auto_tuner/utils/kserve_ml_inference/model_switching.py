import asyncio
import aiohttp

from auto_tuner.utils.kube_resources.services import get_endpoints
from auto_tuner.utils.kube_resources.configmaps import update_configmap
from .constants import CURRENT_MODEL_KEY


class ModelSwitchingMixin:
    def select_model(self, model_name):
        return getattr(self.models, model_name)(pretrained=True)

    def switch_model(self, new_model: str):
        self.model = self.select_model(new_model)


def switch_model(configmap_name: str, namespace: str, service_name: str, target_port: int, new_model: str):

    async def request_pod_to_switch(session, endpoint):
        async with session.post(
                f"{endpoint}/v1/models/{service_name}:switch-model", data={CURRENT_MODEL_KEY: new_model}
        ) as response:
            return await response.json()

    async def request_all_pods_to_switch():
        endpoints = get_endpoints(f"{service_name}-predictor-default", target_port, namespace=namespace)
        tasks = []
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                tasks.append(request_pod_to_switch(session, endpoint))
            responses = await asyncio.gather(*tasks)
            for r in responses:
                # Todo: check if all returned OK
                print(r)

    update_configmap(configmap_name, namespace=namespace, data={CURRENT_MODEL_KEY: new_model}, partial=True)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(request_all_pods_to_switch())
