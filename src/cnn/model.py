import keras
import pandas as pd
from keras import layers

from config import Config


def _conv_block(x, filters: int, kernel: int, stride: int, cfg: Config,
                name: str):
    x = layers.Conv2D(
        filters, kernel, strides=stride, padding="same", use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(cfg.weight_decay),
        name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def build_model(cfg: Config) -> keras.Model:
    inputs = keras.Input(shape=cfg.input_shape, name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)

    x = _conv_block(x, cfg.stem_filters, 5, 2, cfg, "stem")
    if cfg.spatial_dropout[0] > 0:
        x = layers.SpatialDropout2D(cfg.spatial_dropout[0], name="stem_sdrop")(x)

    for i, filters in enumerate(cfg.stage_filters, start=1):
        x = _conv_block(x, filters, 3, 1, cfg, f"stage{i}a")
        x = _conv_block(x, filters, 3, 1, cfg, f"stage{i}b")
        x = layers.MaxPooling2D(2, strides=2, name=f"stage{i}_pool")(x)
        rate = cfg.spatial_dropout[i]
        if rate > 0:
            x = layers.SpatialDropout2D(rate, name=f"stage{i}_sdrop")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(
        cfg.dense_units, use_bias=False, kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(cfg.weight_decay),
        name="head_dense")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Activation("relu", name="head_relu")(x)
    x = layers.Dropout(cfg.dropout, name="head_drop")(x)
    outputs = layers.Dense(cfg.num_classes, activation="softmax",
                           dtype="float32", name="probs")(x)

    return keras.Model(inputs, outputs, name="ham10000_cnn")


def compile_model(model: keras.Model, cfg: Config) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            clipnorm=cfg.clipnorm),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg.label_smoothing),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="acc"),
            keras.metrics.F1Score(average="macro", name="macro_f1"),
        ],
    )
    return model


def receptive_field_report(cfg: Config) -> pd.DataFrame:
    spec = [("stem_conv", 5, 2)]
    for i in range(1, len(cfg.stage_filters) + 1):
        spec += [(f"stage{i}a_conv", 3, 1), (f"stage{i}b_conv", 3, 1),
                 (f"stage{i}_pool", 2, 2)]

    rf, jump, size = 1, 1, cfg.image_size
    rows = []
    for name, kernel, stride in spec:
        rf += (kernel - 1) * jump
        jump *= stride
        size = -(-size // stride)
        rows.append({"layer": name, "kernel": kernel, "stride": stride,
                     "output": f"{size}x{size}", "jump": jump,
                     "receptive_field": rf,
                     "rf_coverage_%": round(min(rf, cfg.image_size) /
                                            cfg.image_size * 100, 1)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = Config()
    model = compile_model(build_model(cfg), cfg)
    model.summary()
    print()
    print(receptive_field_report(cfg).to_string(index=False))
