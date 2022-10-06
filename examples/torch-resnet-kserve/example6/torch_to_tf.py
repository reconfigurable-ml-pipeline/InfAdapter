# Adapted from https://github.com/jimmyyhwu/resnet18-tf2


from multiprocessing import Process
import torchvision.models as models

import tensorflow as tf
from tensorflow.keras import layers

import models as resnet_models


def target(v):
    resnet_model = getattr(resnet_models, f"resnet{v}")

    torch_model = getattr(models, f"resnet{v}")(pretrained=True)
    torch_model.eval()


    # tf.keras.backend.set_floatx('float64')
    inputs = tf.keras.Input(shape=(None, None, 3))
    outputs = resnet_model(inputs)
    model = tf.keras.Model(inputs, outputs)

    #print(torch_model)
    #print(model.summary())

    state_dict = torch_model.state_dict()
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            layer.set_weights([state_dict[f'{layer.name}.weight'].numpy().transpose((2, 3, 1, 0))])
        elif isinstance(layer, layers.Dense):
            layer.set_weights([
                state_dict[f'{layer.name}.weight'].numpy().transpose(),
                state_dict[f'{layer.name}.bias'].numpy()
            ])
        elif isinstance(layer, layers.BatchNormalization):
            keys = ['weight', 'bias', 'running_mean', 'running_var']
            layer.set_weights([state_dict[f'{layer.name}.{key}'].numpy() for key in keys])


    tf.saved_model.save(model, f"tensorflow/{v}")
    print("Exporter version", v)


versions = [18, 34, 50, 101, 152]
processes = []
for version in versions:
    p = Process(target=target, args=(version,))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
