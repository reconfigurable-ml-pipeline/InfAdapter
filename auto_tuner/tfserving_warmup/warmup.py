import os
import time
import grpc
import tensorflow as tf
from tensorflow.python.framework import tensor_util
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


# IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'
warmup_count = int(os.getenv("WARMUP_COUNT", 5))


def get_image_bytes():
    with open("cat.jpg", "rb") as f:
        content = f.read()
    return content


def main():
    image_bytes = get_image_bytes()
    channel = grpc.insecure_channel("127.0.0.1:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    predict_request = predict_pb2.PredictRequest()
    predict_request.model_spec.name = 'resnet'
    predict_request.model_spec.signature_name = 'serving_default'
    predict_request.inputs['b64'].CopyFrom(tensor_util.make_tensor_proto([image_bytes], tf.string))        
        
    x = 0
    while x < warmup_count:
        try:
            response = stub.Predict(predict_request)
            outputs = response.outputs["fc"].float_val
            # print(numpy.argmax(outputs))
        except Exception:
            pass
        else:
            x += 1
    print(f"Sent {warmup_count} warmup requests")
    os.system("touch /warmup/done")
    while True:
        time.sleep(60)
        

if __name__ == "__main__":
    time.sleep(2)
    main()