# Adapted from https://github.com/jimmyyhwu/resnet18-tf2


import math
from tensorflow import keras
from tensorflow.keras import layers

kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

def conv3x3(x, out_planes, stride=1, name=None):
    x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
    return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=name)(x)

def conv1x1(x, out_planes, stride=1, name=None):
    return layers.Conv2D(filters=out_planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=name)(x)


def basic_block(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
    out = layers.ReLU(name=f'{name}.relu1')(out)

    out = conv3x3(out, planes, name=f'{name}.conv2')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = layers.Add(name=f'{name}.add')([identity, out])
    out = layers.ReLU(name=f'{name}.relu2')(out)

    return out

def bottleneck_block(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out = conv1x1(x, planes, name=f'{name}.conv1')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
    out = layers.ReLU(name=f'{name}.relu1')(out)

    out = conv3x3(out, planes, stride=stride, name=f'{name}.conv2')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)
    out = layers.ReLU(name=f'{name}.relu2')(out)

    out = conv1x1(out, planes * 4, name=f'{name}.conv3')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn3')(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = layers.Add(name=f'{name}.add')([identity, out])
    out = layers.ReLU(name=f'{name}.relu3')(out)
    
    return out


inplanes = 64

def make_layer(block, x, planes, blocks, stride=1, name=None):
    global inplanes

    downsample = None
    if stride != 1 or inplanes != planes * (1 if block == basic_block else 4):
        downsample = [
            layers.Conv2D(filters=planes * (1 if block == basic_block else 4), kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=f'{name}.0.downsample.0'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
        ]

    x = block(x, planes, stride, downsample, name=f'{name}.0')
    inplanes = planes * (1 if block == basic_block else 4)
    for i in range(1, blocks):
        x = block(x, planes, name=f'{name}.{i}')

    return x

def resnet(block, x, blocks_per_layer, num_classes=1000):
    global inplanes
    inplanes = 64

    x = layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal, name='conv1')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

    x = make_layer(block, x, 64, blocks_per_layer[0], name='layer1')
    x = make_layer(block, x, 128, blocks_per_layer[1], stride=2, name='layer2')
    x = make_layer(block, x, 256, blocks_per_layer[2], stride=2, name='layer3')
    x = make_layer(block, x, 512, blocks_per_layer[3], stride=2, name='layer4')

    x = layers.GlobalAveragePooling2D(name='avgpool')(x)
    initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
    x = layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)

    return x

def resnet18(x, **kwargs):
    return resnet(basic_block, x, [2, 2, 2, 2], **kwargs)

def resnet34(x, **kwargs):
    return resnet(basic_block, x, [3, 4, 6, 3], **kwargs)

def resnet50(x, **kwargs):
    return resnet(bottleneck_block, x, [3, 4, 6, 3], **kwargs)


def resnet101(x, **kwargs):
    return resnet(bottleneck_block, x, [3, 4, 23, 3], **kwargs)


def resnet152(x, **kwargs):
    return resnet(bottleneck_block, x, [3, 8, 36, 3], **kwargs)

