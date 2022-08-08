import asyncio
import numpy as np 
from aiohttp import ClientSession
from auto_tuner import AUTO_TUNER_DIRECTORY
import time


images = list(
    np.load(f"{AUTO_TUNER_DIRECTORY}/experiments/saved_inputs.npy", allow_pickle=True)
)

async def predict(url, data, delay, session):
    await asyncio.sleep(delay)
    t = time.perf_counter()
    async with session.post(url, data=data["data"]) as response:
        response = await response.json()
        
        #idx = np.argmax(prediction)
        #print("Actual:", code_to_label[data["label_code"]])
        # t5 = (-np.array(prediction)).argsort()[:5]
        # i = 1
        # correct_top5 = False
        # for idx in t5:
        #     #print(f"prediction{i}:", idx_to_label[str(idx)])
        #     if idx_to_label[str(idx)] == code_to_label[data["label_code"]]:
        #         correct_top5 = True
        #         break
        #     i += 1
        # print("correct in top 5:", "Yes" if correct_top5 else "No")

        # #print("------------------------------------------")

        return time.perf_counter() - t


async def generate_load_for_second(url, count):
    # Returns total_time, average, min and max latency of the requests
    tasks = []
    delays = np.cumsum(np.random.exponential(1/(count * 1.5), count))
    print("delay of the last request", delays[-1])
    async with ClientSession() as session:
        for i in range(count):
            data = images[np.random.randint(0, 200)]
            task = asyncio.ensure_future(predict(url, data, delays[i], session))
            tasks.append(task)
    
        # print("size", sys.getsizeof(data["data"]) / 1000000)
        t = time.perf_counter()
        rts = await asyncio.gather(*tasks)
        return {
            "total_time": time.perf_counter() - t, 
            "avg": sum(rts) / len(rts),
            "min": min(rts),
            "max": max(rts),
            "p50": np.percentile(rts, 50),
            "p95": np.percentile(rts, 95),
            "p99": np.percentile(rts, 99),
        }, rts


def generate_load(url, total):
    return asyncio.run(generate_load_for_second(url, total))
