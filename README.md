# TP Integrador — Redes Neuronales (CNN + RNN)

Inteligencia Artificial — Lic. en Sistemas de Información.

- **CNN** (`src/cnn`): clasificación de lesiones de piel con HAM10000.
- **RNN** (`src/rnn`): pronóstico de temperatura con Jena Climate.

## Instalación

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

Los scripts se ejecutan desde la carpeta de cada módulo y con el Python del entorno virtual.

## CNN

```bash
cd src/cnn
../../.venv/Scripts/python data_download.py    # descarga y prepara el dataset
../../.venv/Scripts/python train.py            # entrena con Group K-Fold
../../.venv/Scripts/python evaluate.py         # evalúa sobre el conjunto de prueba
```

Análisis opcionales:

```bash
../../.venv/Scripts/python leakage_study.py    # partición aleatoria vs. por lesión
../../.venv/Scripts/python learning_curve.py   # curva de aprendizaje
```

## RNN

```bash
cd src/rnn
../../.venv/Scripts/python data_download.py              # descarga y prepara el dataset
../../.venv/Scripts/python train.py --cell lstm          # entrena la LSTM
../../.venv/Scripts/python evaluate.py --cell lstm       # evalúa sobre el conjunto de prueba
```

Análisis opcional:

```bash
../../.venv/Scripts/python gradient_study.py             # comparación de celdas en BPTT
```

## Modelos entrenados

Los pesos quedan en `models/` en formato `.keras` y se cargan directamente:

```python
import keras
model = keras.models.load_model("models/rnn/jena_lstm.keras")
```

- `models/cnn/group_fold0.keras` … `group_fold4.keras`: CNN, un modelo por partición.
- `models/rnn/jena_lstm.keras`: LSTM.

Los resultados y figuras de la evaluación se guardan en `reports/`.
