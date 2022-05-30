import onnx
from tensorflow.python.tools.import_pb_to_tensorboard import import_to_tensorboard
from onnx_tf.backend import prepare

versions = [18, 34, 50, 101, 152]

for v in versions:
    onnx_model = onnx.load(f"onnx/resnet{v}.onnx")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(f"tensorflow/{v}")
