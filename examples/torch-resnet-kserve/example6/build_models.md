1. Install tensorflow
    ```shell
    pip install tensorflow==2.9.1
    pip install tensorflow_probability==0.17.0
    ```

2. Install pytorch
    ```shell
    pip install torch==1.10.1
    pip install torchvision==0.11.2
    ```

3. Convert Pytorch pretrained resnet models to Tensorflow saved model
    ```shell
    python torch_to_tf.py
    ```

4. Add preprocessing to the generated Tensorflow models to allow them accept base64 image as input
    ```shell
    mkdir tensorflow_b64
    python tensorflow_b64.py
    ```
5. mkdir -p /fileshare/tensorflow_resnet_b64
6. cp -r tensorflow_b64/* /fileshare/tensorflow_resnet_b64