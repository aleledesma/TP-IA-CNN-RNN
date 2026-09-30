import keras
from keras import layers

from config import Config

CELLS = {
    "simple": layers.SimpleRNN,
    "lstm": layers.LSTM,
    "gru": layers.GRU,
}


def build_model(cfg: Config) -> keras.Model:
    if cfg.cell not in CELLS:
        raise ValueError(f"unknown cell '{cfg.cell}', expected {list(CELLS)}")
    cell = CELLS[cfg.cell]

    inputs = keras.Input(shape=cfg.input_shape, name="window")
    x = inputs
    for i, units in enumerate(cfg.units):
        last = i == len(cfg.units) - 1
        x = cell(units, return_sequences=not last,
                 dropout=cfg.dropout,
                 recurrent_dropout=cfg.recurrent_dropout,
                 kernel_regularizer=keras.regularizers.l2(cfg.weight_decay),
                 name=f"{cfg.cell}{i + 1}")(x)
    x = layers.Dropout(cfg.dropout, name="head_drop")(x)
    outputs = layers.Dense(cfg.horizon, dtype="float32", name="forecast")(x)

    return keras.Model(inputs, outputs, name=f"jena_{cfg.cell}")


def compile_model(model: keras.Model, cfg: Config) -> keras.Model:
    loss = keras.losses.Huber(delta=1.0) if cfg.loss == "huber" else cfg.loss
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=cfg.learning_rate,
                                         weight_decay=cfg.weight_decay,
                                         clipnorm=cfg.clipnorm),
        loss=loss,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae"),
                 keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def parameter_report(cfg: Config) -> dict:
    report = {}
    for name in CELLS:
        variant = Config(**{**cfg.__dict__, "cell": name})
        model = build_model(variant)
        report[name] = {
            "params": int(model.count_params()),
            "gates": {"simple": 1, "gru": 3, "lstm": 4}[name],
        }
        keras.backend.clear_session()
    return report


if __name__ == "__main__":
    cfg = Config()
    compile_model(build_model(cfg), cfg).summary()
    print()
    for name, info in parameter_report(cfg).items():
        print(f"{name:7s} gates={info['gates']} params={info['params']:,}")
