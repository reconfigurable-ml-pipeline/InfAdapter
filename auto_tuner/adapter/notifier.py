import requests
import time
import os




if __name__ == "__main__":
    time.sleep(60 * int(os.environ["FIRST_DECIDE_DELAY_MINUTES"]))
    while True:
        requests.post("http://localhost:8000/decide", timeout=30)
        time.sleep(60)
