import requests
import time



if __name__ == "__main__":
    time.sleep(60 * 11)
    while True:
        requests.post("http://localhost:8000/decide", timeout=30)
        time.sleep(60)
