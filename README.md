# CRNN Text Recognition

OCR-система для распознавания строк текста на базе CRNN (CNN + BiLSTM + CTC).

Проект включает:
- обучение модели;
- генерацию synthetic dataset;
- inference и evaluation;
- FastAPI web API;
- Docker deployment;
- Jupyter experiments.

---

# Архитектура

Модель построена по схеме:

```text
Image
  ↓
CNN feature extractor
  ↓
BiLSTM
  ↓
CTC decoder
  ↓
Predicted text
```

Используется:
- PyTorch;
- CRNN architecture;
- CTC Loss;
- dynamic-width OCR pipeline.

---

# Возможности

- OCR для строк текста;
- поддержка динамической ширины изображений;
- confidence score;
- CER/WER evaluation;
- visualization prediction tools;
- hard-example analysis;
- checkpoint resume;
- FastAPI inference API;
- Docker support.

---

# Стек

- Python 3.11
- PyTorch
- torchvision
- FastAPI
- Uvicorn
- Pillow
- matplotlib
- pandas

---

# Установка

## 1. Клонирование

```bash
git clone <repository-url>
cd crnn-text-recognition
```

---

## 2. Virtual environment

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

---

# Dataset

Проект поддерживает:
- MJSynth;
- собственный synthetic dataset;
- TextRecognitionDataGenerator.

---

## Генерация synthetic dataset

```bash
python scripts/generate_gendata.py \
  --count 100000 \
  --mode dict \
  --language en \
  --words 5 \
  --font-size 32 \
  --distortion 2 \
  --clean
```

Результат:

```text
data/processed/gendata/
├── images/
├── labels.csv
└── labels.txt
```

---

# Обучение

## Запуск training

```bash
python scripts/train.py
```

Модель автоматически:
- сохраняет checkpoints;
- сохраняет best model;
- поддерживает resume training.

---

## Конфиг

Основные параметры находятся в:

```text
configs/config.yaml
```

Пример:

```yaml
batch_size: 16
lr: 0.001
epochs: 30
img_height: 48
img_width: 768
```

---

# Inference

## Predict image

```bash
python scripts/predict.py --image path/to/image.png
```

---

# FastAPI Web API

## Запуск

```bash
uvicorn app:app --reload
```

---

## Endpoint

### POST `/predict`

Принимает изображение и возвращает:

```json
{
  "text": "predicted text",
  "confidence": 94.12,
  "success": true
}
```

---

# Docker

## Build image

```bash
docker build -t crnn-ocr .
```

---

## Run container

```bash
docker run -p 8000:8000 crnn-ocr
```

---

# Evaluation

Поддерживается:
- Accuracy;
- CER (Character Error Rate);
- WER (Word Error Rate);
- hard examples analysis;
- prediction visualization.

---

# Структура проекта

```text
crnn-text-recognition/
├── app.py
├── configs/
├── data/
├── notebooks/
├── outputs/
├── scripts/
├── src/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# Основные модули

```text
src/data.py
```
OCR dataset + transforms.

```text
src/model.py
```
CRNN architecture.

```text
src/text_codec.py
```
CTC encode/decode.

```text
src/inference.py
```
Inference + confidence.

```text
src/train.py
```
Training loop + checkpoints.

```text
src/evaluation.py
```
Visualization + metrics.

---

# Notebook

Основной notebook:

```text
notebooks/experiments.ipynb
```

Используется для:
- experiments;
- evaluation;
- visualization;
- debugging.

---

# Текущие особенности модели

- dynamic-width OCR;
- RGB pipeline;
- CTC decoding;
- synthetic dataset training;
- adaptive batch padding;
- checkpoint resume support.

---

# Автор

Begzad