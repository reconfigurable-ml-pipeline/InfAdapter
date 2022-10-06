from multiprocessing import Process
import tensorflow as tf


def convert(v):
    model = tf.saved_model.load(f"tensorflow/{v}")
    # Current signature, accepting tensor with shape (-1, -1, -1, 3)
    signature = model.signatures["serving_default"]

    @tf.function()
    def predict_b64(image_b64):
        def preprocess(image_b64):
            img = tf.image.decode_jpeg(image_b64, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, (224, 224))
            # img = tf.transpose(img, perm=[2, 0, 1])
            return img
        img_tensor = tf.nest.map_structure(
            tf.stop_gradient,
            tf.map_fn(lambda x: preprocess(x), elems=image_b64, fn_output_signature=tf.float32)
        )
        prediction = signature(img_tensor)
        return prediction


    new_signature = predict_b64.get_concrete_function(
        image_b64=tf.TensorSpec([None], dtype=tf.string, name="b64")
    )
    tf.saved_model.save(
        model,
        export_dir=f"tensorflow_b64/{v}",
        signatures=new_signature
    )
    print("Saved version", v)


if __name__ == "__main__":
    versions = [18, 34, 50, 101, 152]
    processes = []

    for version in versions:
        p = Process(target=convert, args=(version,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
