from multiprocessing import Process
import onnx
from onnx_tf.backend import prepare

versions = [18, 34, 50, 101, 152]

def convert(v):
    onnx_model = onnx.load(f"onnx/resnet{v}.onnx")
    #onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
    tf_rep = prepare(onnx_model, logging_level="DEBUG")
    tf_rep.export_graph(f"tensorflow/{v}")
    print("Exported version", v)

ps = []
for v in versions:
    p = Process(target=convert, args=(v,))
    p.start()
    ps.append(p)


for proc in ps:
    proc.join()