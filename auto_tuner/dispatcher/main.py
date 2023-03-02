import os
from aiohttp import ClientSession, web, ClientTimeout


class Dispatcher:
    def __init__(self) -> None:
        self.model_names = []
        self.endpoints = []
        self.quotas = {}
        self.url_path = os.getenv("URL_PATH", "/")
        self.r = 1
        self.idx = 0
        self.total_requests = {}
        self.service_names = {}
        self.sessions = {}
        
    
    def initialize(self, data: dict):
        self.model_names = list(data.keys())
        self.endpoints = list(data.values())
        for model_name in self.model_names:
            self.sessions[model_name] = ClientSession(timeout=ClientTimeout(total=int(os.getenv("TIMEOUT", 30))))
            self.total_requests[model_name] = 0
    
    
    def reset(self, data: dict):
        self.quotas: dict = data["quotas"]
        self.r = 1
        self.idx = 0
    
    
    async def dispatch(self, data):
        endpoint = None
        chosen_model = None
        while True:
            quota = self.quotas[self.model_names[self.idx]]
            if quota >= self.r:
                endpoint = self.endpoints[self.idx]
                chosen_model = self.model_names[self.idx]
            self.idx += 1
            if self.idx == len(self.model_names):
                self.idx = 0
                self.r += 1
                if self.r > max(self.quotas.values()):
                    self.r = 1
            if endpoint:
                break
        
        self.total_requests[chosen_model] += 1
        session = self.sessions[self.model_names[self.idx]]
        async with session.post(f"http://{endpoint}{self.url_path}", data=data) as response:
            response = await response.text()
            return response


dispatcher = Dispatcher()


async def initialize(request):
    data = await request.json()
    if not dispatcher.model_names:
        dispatcher.initialize(data)
        return web.json_response({"success": True})
    return web.json_response({"success": False, "message": "Already initialized."})


async def predict(request):
    data = await request.text()
    return web.Response(body = await dispatcher.dispatch(data))


async def reset(request):
    data = await request.json()
    dispatcher.reset(data)
    return web.json_response({"success": True})


async def export_request_count(request):
    content = "# HELP dispatcher_requests_total Total number of requests labeled by model used for inference.\n"
    content += "# TYPE dispatcher_requests_total counter\n"
    for model, counter in dispatcher.total_requests.items():
        content += f'dispatcher_requests_total{{model="{model}"}} {counter}\n'
    return web.Response(body=content)


app = web.Application()
app.add_routes(
    [
        web.post("/initialize", initialize),
        web.post("/reset", reset),
        web.post("/predict", predict),
        web.get("/metrics", export_request_count),
    ]
)
if __name__ == '__main__':
    web.run_app(app, host="0.0.0.0", port=8000, access_log=None)
