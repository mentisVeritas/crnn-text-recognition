# CRNN Text Recognition

Реализация модели CRNN (Convolutional Recurrent Neural Network) для распознавания текста на изображениях (OCR).

## Описание

Этот проект реализует модель CRNN для распознавания текста на изображениях. Модель сочетает сверточные нейронные сети для извлечения признаков и рекуррентные нейронные сети для последовательного распознавания текста.

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone <repository-url>
   cd crnn-text-recognition
   ```

2. Создайте виртуальное окружение:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # для Linux/Mac
   # или .venv\Scripts\activate для Windows
   ```

3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Датасет

Используется датасет MJSynth-1-250k с Kaggle:

- **Ссылка**: https://www.kaggle.com/datasets/pradeepsiva/mjsynth-1-250k

### Скачивание и подготовка данных

1. Перейдите в папку данных:
   ```bash
   cd data/raw
   ```

2. Скачайте датасет:
   ```bash
   kaggle datasets download pradeepsiva/mjsynth-1-250k
   ```

3. Распакуйте:
   ```bash
   unzip mjsynth-1-250k.zip
   rm mjsynth-1-250k.zip
   ```

### Структура данных

```
data/raw/
├── imgs1_w/
│   ├── imgs1/
│   │   └── *.png  # изображения
│   └── imgs1_text_recog.csv  # разметка
```

### Разметка

Разметка хранится в CSV файле `imgs1_text_recog.csv` со столбцами:
- `image_name` — имя файла изображения (в папке `imgs1/`)
- `utf8string` — текст на изображении

## Использование

### Обучение модели

Запустите скрипт обучения:
```bash
python scripts/train.py
```

Сообщения уровня INFO (и выше) из корневого логгера дополнительно **дописываются в файл** `outputs/logs/train.log` (папка создаётся автоматически; то же при обучении из ноутбука с `TRAINING=True`, если путь к чекпоинту вида `.../outputs/checkpoints/...`).

### Предсказание

Запустите скрипт предсказания (нужен чекпоинт `outputs/checkpoints/best_model.pth` после обучения):
```bash
python scripts/predict.py --image path/to/image.png
```

Без `--image` скрипт возьмёт до пяти случайных `.png` из папки изображений из конфига.

### Ноутбук

Основной интерактивный сценарий: `notebooks/experiments.ipynb`.

В ноутбуке оставлен orchestration (запуск шагов, сравнение, визуальный контроль), а переиспользуемая логика вынесена в `src/`:
- `src/train.py` — setup + цикл обучения + чекпоинты + `run_training`
- `src/data.py` — датасет OCR + общий image transform builder
- `src/text_codec.py` — encode/decode текста для CTC
- `src/inference.py` — preprocessing + предсказание с confidence
- `src/evaluation.py` — визуализация, hard examples, leaderboard и лог экспериментов
- `src/metrics.py` — Levenshtein, Accuracy/CER/WER
- `src/visualization.py` — общий рендер prediction grid
- `src/utils.py` — конфиг, устройство и runtime/logging helper'ы

## Структура проекта

```
crnn-text-recognition/
├── configs/              # конфигурационные файлы
├── data/                 # данные
│   ├── processed/        # обработанные данные
│   └── raw/              # сырые данные
├── models/               # сохраненные модели
├── notebooks/            # Jupyter ноутбуки
├── scripts/              # скрипты для запуска
│   ├── predict.py        # скрипт предсказания
│   └── train.py          # скрипт обучения
├── src/                  # исходный код
│   ├── data.py           # датасет OCR + image transform builder
│   ├── model.py          # модель CRNN
│   ├── text_codec.py     # encode/decode текста для CTC
│   ├── inference.py      # инференс + confidence
│   ├── train.py          # setup + цикл обучения + чекпоинты + run_training
│   ├── evaluation.py     # визуализация, error analysis, leaderboard + experiment log
│   ├── metrics.py        # Accuracy/CER/WER + edit distance
│   ├── visualization.py  # визуализация предсказаний
│   └── utils.py          # конфиг, устройство и runtime/logging helpers
├── requirements.txt      # зависимости
└── README.md             # этот файл
```

## Лицензия

[Укажите лицензию, если применимо]
