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

### Предсказание

Запустите скрипт предсказания (нужен чекпоинт `outputs/checkpoints/best_model.pth` после обучения):
```bash
python scripts/predict.py --image path/to/image.png
```

Без `--image` скрипт возьмёт до пяти случайных `.png` из папки изображений из конфига.

### Ноутбук

Основной интерактивный сценарий: `notebooks/experiments.ipynb`.

В ноутбуке оставлен orchestration (запуск шагов, сравнение, визуальный контроль), а переиспользуемая логика вынесена в `src/`:
- `src/notebook_runner.py` — полный пайплайн для `experiments.ipynb` (dataloaders, train loop, визуализация, лог)
- `src/inference.py` — preprocessing + предсказание с confidence
- `src/metrics.py` — Levenshtein, Accuracy/CER/WER
- `src/visualization.py` — общий рендер prediction grid
- `src/experiment_log.py` — сбор предсказаний и лог запусков

## Структура проекта

```
crnn-text-recognition/
├── configs/              # конфигурационные файлы
├── data/                 # данные
│   ├── processed/        # обработанные данные
│   └── raw/              # сырые данные
├── logs/                 # логи обучения
├── models/               # сохраненные модели
├── notebooks/            # Jupyter ноутбуки
├── scripts/              # скрипты для запуска
│   ├── predict.py        # скрипт предсказания
│   └── train.py          # скрипт обучения
├── src/                  # исходный код
│   ├── data.py           # датасет OCR (CSV + изображения)
│   ├── model.py          # модель CRNN
│   ├── train.py          # цикл обучения (CTC)
│   ├── decode.py         # жадное декодирование CTC
│   ├── inference.py      # инференс + confidence
│   ├── metrics.py        # Accuracy/CER/WER + edit distance
│   ├── visualization.py  # визуализация предсказаний
│   ├── experiment_log.py # лог/leaderboard экспериментов
│   ├── notebook_runner.py # логика ноутбука experiments (train / eval / viz)
│   └── utils.py          # конфиг и устройство
├── requirements.txt      # зависимости
└── README.md             # этот файл
```

## Лицензия

[Укажите лицензию, если применимо]
