import generic_train
import tensorflow as tf


def vgginnet_builder(image_size=224, num_classes=3):
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        input_shape=(image_size, image_size, 3),
        weights='imagenet',
    )

    layer_name = "block4_pool"
    feature_ex_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(layer_name).output,
        name="vgg16_features",
    )
    feature_ex_model.trainable = True

    def naive_inception_module(layer_in, f1, f2, f3):
        # 1x1 conv
        conv1 = tf.keras.layers.Conv2D(f1, (1, 1),
            padding="same", activation="relu")(layer_in)
        # 3x3 conv
        conv3 = tf.keras.layers.Conv2D(f2, (3, 3),
            padding="same", activation="relu")(layer_in)
        # 5x5 conv
        conv5 = tf.keras.layers.Conv2D(f3, (5, 5),
            padding="same", activation="relu")(layer_in)
        # 3x3 max pooling
        pool = tf.keras.layers.MaxPooling2D((3, 3),
            strides=(1, 1), padding="same")(layer_in)
        # Concatenate filters, assumes filters/channels last
        layer_out = tf.keras.layers.Concatenate()([
            conv1, conv3, conv5, pool])
        return layer_out

    out = naive_inception_module(feature_ex_model.output, 64, 128, 32)

    bn1 = tf.keras.layers.BatchNormalization(name="bn1")(out)

    f = tf.keras.layers.Flatten()(bn1)
    dropout = tf.keras.layers.Dropout(0.4, name="dropout")(f)
    desne = tf.keras.layers.Dense(num_classes, 
        activation="softmax", 
        name="predictions")(dropout)

    model = tf.keras.Model(inputs=feature_ex_model.input, outputs=desne)
    return model


def vgg_model_fn():
    copy_model = tf.keras.applications.vgg16.VGG16(
        include_top=True, weights="imagenet", pooling=None,
    )
    copy_model.trainable = True

    model = tf.keras.Sequential([
        *(copy_model.layers[:-1]),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])

    model.build([None, 224, 224, 3])
    return model


def generate_model_fn(model_fn):
    def generated_model_fn():
        model = model_fn(
            include_top=False,
            input_shape=[224, 224, 3],
            weights='imagenet',
            pooling='avg'
        )
        model.trainable = True
        model = tf.keras.Sequential([
            model,
            tf.keras.layers.Dense(3, activation="softmax"),
        ])
        return model
    return generated_model_fn


experiments = [
    lambda: generic_train.train_on_cervix(
        vgginnet_builder,
        "./cervix/vggin-net-wo-reject-resample-1",
        tf.keras.applications.vgg16.preprocess_input,
        augment=True,
        reject_resample=False,
    ),
    lambda: generic_train.train_on_cervix(
        vgg_model_fn,
        "./cervix/vgg16-wo-reject-resample-1",
        tf.keras.applications.vgg16.preprocess_input,
        augment=True,
        reject_resample=False,
    ),
    lambda: generic_train.train_on_cervix(
        generate_model_fn(tf.keras.applications.InceptionV3),
        "./cervix/inception_v3-wo-reject-resample-1",
        tf.keras.applications.inception_v3.preprocess_input,
        augment=True,
        reject_resample=False,
    ),
    lambda: generic_train.train_on_cervix(
        generate_model_fn(tf.keras.applications.ResNet50),
        "./cervix/resnet50-wo-reject-resample-1",
        tf.keras.applications.resnet50.preprocess_input,
        augment=True,
        reject_resample=False,
    ),
    lambda: generic_train.train_on_cervix(
        generate_model_fn(tf.keras.applications.xception.Xception),
        "./cervix/xception-wo-reject-resample-1",
        tf.keras.applications.xception.preprocess_input,
        augment=True,
        reject_resample=False,
    ), 
    lambda: generic_train.train_on_cervix(
        generate_model_fn(tf.keras.applications.InceptionResNetV2),
        "./cervix/inception_resnet_v2-wo-reject-resample-1",
        tf.keras.applications.inception_resnet_v2.preprocess_input,
        augment=True,
        reject_resample=False,
    ),
    lambda: generic_train.train_on_cervix(
        generate_model_fn(tf.keras.applications.DenseNet121),
        "./cervix/densenet121-wo-reject-resample-1",
        tf.keras.applications.densenet.preprocess_input,
        augment=True,
        reject_resample=False,
    ),
    lambda: generic_train.train_on_cervix(
        generate_model_fn(tf.keras.applications.EfficientNetB0),
        "./cervix/efficientnet-b0-wo-reject-resample-1",
        tf.keras.applications.efficientnet.preprocess_input,
        augment=True,
        reject_resample=False,
    ),
    lambda: generic_train.train_on_cervix(
        generate_model_fn(tf.keras.applications.ResNet50V2),
        "./cervix/resnet50v2-wo-reject-resample-1",
        tf.keras.applications.resnet_v2.preprocess_input,
        augment=True,
        reject_resample=False,
    ),

    lambda: generic_train.train_on_cervix(
        vgg_model_fn,
        "./cervix/vgg16-wo-data-augment-1",
        tf.keras.applications.vgg16.preprocess_input,
        augment=False,
        reject_resample=True,
    ),
    lambda: generic_train.train_on_cervix(
        generate_model_fn(tf.keras.applications.InceptionV3),
        "./cervix/inception_v3-wo-data-augment-1",
        tf.keras.applications.inception_v3.preprocess_input,
        augment=False,
        reject_resample=True,
    ),
    lambda: generic_train.train_on_cervix(
        generate_model_fn(tf.keras.applications.ResNet50),
        "./cervix/resnet50-wo-data-augment-1",
        tf.keras.applications.resnet50.preprocess_input,
        augment=False,
        reject_resample=True,
    ),
    lambda: generic_train.train_on_cervix(
        generate_model_fn(tf.keras.applications.xception.Xception),
        "./cervix/xception-wo-data-augment-1",
        tf.keras.applications.xception.preprocess_input,
        augment=False,
        reject_resample=True,
    ),
    lambda: generic_train.train_on_cervix(
        generate_model_fn(tf.keras.applications.InceptionResNetV2),
        "./cervix/inception_resnet_v2-wo-data-augment-1",
        tf.keras.applications.inception_resnet_v2.preprocess_input,
        augment=False,
        reject_resample=True,
    ),
    lambda: generic_train.train_on_cervix(
        generate_model_fn(tf.keras.applications.DenseNet121),
        "./cervix/densenet121-wo-data-augment-1",
        tf.keras.applications.densenet.preprocess_input,
        augment=False,
        reject_resample=True,
    ),
    lambda: generic_train.train_on_cervix(
        generate_model_fn(tf.keras.applications.EfficientNetB0),
        "./cervix/efficientnet-b0-wo-data-augment-1",
        tf.keras.applications.efficientnet.preprocess_input,
        augment=False,
        reject_resample=True,
    ),
    lambda: generic_train.train_on_cervix(
        generate_model_fn(tf.keras.applications.ResNet50V2),
        "./cervix/resnet50v2-wo-data-augment-1",
        tf.keras.applications.resnet_v2.preprocess_input,
        augment=False,
        reject_resample=True,
    ),
    lambda: generic_train.train_on_cervix(
        vgginnet_builder,
        "./cervix/vggin-net-wo-data-augment-1",
        tf.keras.applications.vgg16.preprocess_input,
        augment=False,
        reject_resample=True,
    ),
]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train various models on the cervix dataset.')
    parser.add_argument('-c', '--choice', type=int, required=True, help='choice of model')
    args = parser.parse_args()
    experiments[args.choice]()
