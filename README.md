# Система отслеживания и сегментации движений глаз

Комплексная Python система для детектирования движений зрачков, отслеживания взгляда и сегментации движений глаз на фиксации и саккады с использованием нескольких алгоритмов.

## Содержание

1. [Установка](#установка)
2. [Структура проекта](#структура-проекта)
3. [Быстрый старт](#быстрый-старт)
4. [Детальное описание классов](#детальное-описание-классов)
5. [Форматы данных](#форматы-данных)
6. [Примеры использования](#примеры-использования)
7. [Алгоритмы сегментации](#алгоритмы-сегментации)
8. [Конфигурация параметров](#конфигурация-параметров)

## Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/your-username/Attention-deficit-disorder.git
cd Attention-deficit-disorder
```

### 2. Установка зависимостей

```bash
# Создание виртуального окружения (рекомендуется)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# Установка пакетов
pip install -r requirements.txt
```

### 3. Установка проекта

```bash
pip install -e .
```

### Зависимости

- **opencv-python** >= 4.5.0 - обработка изображений и видео
- **mediapipe** >= 0.8.0 - детектирование лица и зрачков
- **numpy** >= 1.21.0 - работа с массивами
- **pandas** >= 1.3.0 - работа с данными
- **scipy** >= 1.7.0 - математические функции
- **hmmlearn** >= 0.2.7 - алгоритм HMM
- **plotly** >= 5.0.0 - интерактивные графики

## Структура проекта

```
Attention-deficit-disorder/
├── src/
│   ├── detection/          # Детектирование зрачков
│   │   ├── __init__.py
│   │   └── pupil_detector.py
│   ├── segmentation/       # Алгоритмы сегментации
│   │   ├── __init__.py
│   │   ├── ivt.py         # I-VT алгоритм
│   │   ├── idt.py         # I-DT алгоритм
│   │   └── hmm.py         # HMM алгоритм
│   ├── visualization/      # Визуализация
│   │   ├── __init__.py
│   │   └── plotter.py
│   └── utils/             # Утилиты
│       └── __init__.py
├── examples/              # Примеры использования
│   └── main_example.py
├── scripts/               # Вспомогательные скрипты
├── tests/                 # Тесты
├── data/                  # Данные
├── requirements.txt       # Зависимости
├── setup.py              # Установка пакета
└── README_RU.md          # Эта документация
```

## Быстрый старт

### Пример 1: Детектирование зрачков из видео

```python
from src.detection import PupilDetector

# Создание детектора
detector = PupilDetector()

# Обработка видеофайла
detector.process_video('path/to/video.mp4', output_csv='pupil_data.csv')

print("Детектирование завершено! Результаты сохранены в pupil_data.csv")
```

### Пример 2: Сегментация движений глаз

```python
from src.segmentation import IVTSegmenter

# Создание сегментатора
segmenter = IVTSegmenter(velocity_threshold=110)

# Сегментация движений
result = segmenter.segment('pupil_data.csv')

# Сохранение результатов
result['fixations'].to_csv('fixations.csv', index=False)
result['saccades'].to_csv('saccades.csv', index=False)
```

### Пример 3: Визуализация

```python
from src.visualization import EyeMovementPlotter

# Создание построителя графиков
plotter = EyeMovementPlotter()

# Построение графика с сегментацией
plotter.plot_with_segmentation(
    pupil_csv='pupil_data.csv',
    fixations_csv='fixations.csv',
    saccades_csv='saccades.csv',
    output_file='result.html'
)
```

## Детальное описание классов

### PupilDetector (src/detection/pupil_detector.py)

**Назначение:** Базовый детектор зрачков, использующий MediaPipe Face Mesh.

#### Методы:

##### `detect_pupils_in_frame(frame, frame_id=None)`
- **Описание:** Детектирует зрачки в одном кадре
- **Параметры:**
  - `frame` (np.ndarray): Кадр изображения в формате BGR
  - `frame_id` (str, optional): Идентификатор кадра
- **Возвращает:** Словарь с координатами 'left' и 'right' (x, y)

##### `process_video(video_path, output_csv="pupil_data.csv", show_result=False)`
- **Описание:** Обрабатывает видеофайл и детектирует зрачки в каждом кадре
- **Параметры:**
  - `video_path` (str): Путь к видео или 0 для веб-камеры
  - `output_csv` (str): Путь для сохранения результатов
  - `show_result` (bool): Показывать ли видео в реальном времени

##### `process_image_folder(input_folder, output_csv="pupil_data.csv")`
- **Описание:** Обрабатывает все изображения в папке
- **Параметры:**
  - `input_folder` (str): Путь к папке с изображениями
  - `output_csv` (str): Путь для сохранения результатов

### PupilHeadDetector (src/detection/pupil_detector.py)

**Назначение:** Детектор зрачков с компенсацией движения головы.

**Наследуется от:** PupilDetector

**Дополнительные возможности:**
- Автоматическая компенсация движения головы
- Использует референсные точки лица (переносица, уголки глаз)
- Сохраняет относительные координаты зрачков

### IVTSegmenter (src/segmentation/ivt.py)

**Назначение:** Классификация движений глаз по скорости (I-VT алгоритм).

#### Конструктор:
```python
IVTSegmenter(
    velocity_threshold=110,      # Порог скорости (пикс/с)
    min_fixation_duration=0.1,   # Минимальная длительность фиксации (с)
    fps=30                       # Частота кадров
)
```

#### Методы:

##### `segment(input_csv, output_prefix="eye_movement")`
- **Описание:** Сегментирует движения глаз на фиксации и саккады
- **Параметры:**
  - `input_csv` (str): Путь к CSV с данными о зрачках
  - `output_prefix` (str): Префикс для выходных файлов
- **Возвращает:** Словарь с DataFrames 'fixations' и 'saccades'

### IDTSegmenter (src/segmentation/idt.py)

**Назначение:** Классификация движений глаз по дисперсии (I-DT алгоритм).

#### Конструктор:
```python
IDTSegmenter(
    dispersion_threshold=7,      # Порог дисперсии (пиксели)
    window_duration=0.2,          # Размер окна (с)
    min_fixation_duration=0.1,   # Минимальная длительность фиксации (с)
    fps=30                       # Частота кадров
)
```

### HMMSegmenter (src/segmentation/hmm.py)

**Назначение:** Классификация движений глаз с помощью модели скрытых марковских цепей.

#### Конструктор:
```python
HMMSegmenter(
    n_components=2,              # Количество состояний
    covariance_type='diag',       # Тип ковариации
    n_iter=100,                   # Итераций обучения
    min_fixation_duration=0.1,    # Минимальная длительность (с)
    fps=30                        # Частота кадров
)
```

### EyeMovementPlotter (src/visualization/plotter.py)

**Назначение:** Построение интерактивных графиков для данных о движениях глаз.

#### Методы:

##### `plot_raw_data(pupil_csv, output_file="eye_movement_plot.html")`
- **Описание:** Строит график сырых данных о движении зрачков
- **Параметры:**
  - `pupil_csv` (str): Путь к CSV с данными о зрачках
  - `output_file` (str): Путь к выходному HTML файлу

##### `plot_with_segmentation(pupil_csv, fixations_csv, saccades_csv, output_file)`
- **Описание:** Строит график с отмеченными фиксациями и саккадами
- **Параметры:**
  - `pupil_csv` (str): Путь к CSV с данными о зрачках
  - `fixations_csv` (str): Путь к CSV с фиксациями
  - `saccades_csv` (str): Путь к CSV с саккадами
  - `output_file` (str): Путь к выходному HTML файлу

## Форматы данных

### Входной формат (CSV для детектирования)

Файл содержит данные о зрачках, полученные после обработки видео или изображений:

```csv
filename,left_x,left_y,right_x,right_y
frame_0_00_00_000000.jpg,320,240,480,240
frame_0_00_00_033333.jpg,325,242,485,242
...
```

**Структура:**
- `filename` - имя файла/идентификатор кадра
- `left_x`, `left_y` - координаты левого зрачка в пикселях
- `right_x`, `right_y` - координаты правого зрачка в пикселях

### Выходной формат для PupilHeadDetector

Дополнительно содержит относительные координаты:

```csv
filename,left_x,left_y,right_x,right_y,rel_left_x,rel_left_y,rel_right_x,rel_right_y
...
```

### Формат результатов сегментации (фиксации)

```csv
start_time,end_time,duration,x_center,y_center,amplitude
0.1,0.5,0.4,400,300,0
0.6,1.2,0.6,450,350,0
...
```

**Структура:**
- `start_time` - время начала фиксации (с)
- `end_time` - время окончания фиксации (с)
- `duration` - длительность фиксации (с)
- `x_center` - центр фиксации по X (пиксели)
- `y_center` - центр фиксации по Y (пиксели)
- `amplitude` - амплитуда (для фиксаций = 0)

### Формат результатов сегментации (саккады)

```csv
start_time,end_time,start_x,start_y,end_x,end_y,amplitude
0.5,0.6,400,300,450,350,64.0
1.2,1.3,450,350,380,320,85.4
...
```

**Структура:**
- `start_time` - время начала саккады (с)
- `end_time` - время окончания саккады (с)
- `start_x`, `start_y` - начальная точка саккады
- `end_x`, `end_y` - конечная точка саккады
- `amplitude` - амплитуда саккады в пикселях

## Примеры использования

### Полный pipeline

```python
from src.detection import PupilDetector
from src.segmentation import IVTSegmenter, IDTSegmenter, HMMSegmenter
from src.visualization import EyeMovementPlotter

# 1. Детектирование зрачков
detector = PupilDetector()
detector.process_video('data/video.mp4', output_csv='pupil_data.csv')

# 2. Сегментация I-VT
ivt = IVTSegmenter(velocity_threshold=110)
ivt_result = ivt.segment('pupil_data.csv')
ivt_result['fixations'].to_csv('fixations_ivt.csv', index=False)
ivt_result['saccades'].to_csv('saccades_ivt.csv', index=False)

# 3. Сегментация I-DT
idt = IDTSegmenter(dispersion_threshold=7)
idt_result = idt.segment('pupil_data.csv')
idt_result['fixations'].to_csv('fixations_idt.csv', index=False)
idt_result['saccades'].to_csv('saccades_idt.csv', index=False)

# 4. Сегментация HMM
hmm = HMMSegmenter(n_iter=100)
hmm_result = hmm.segment('pupil_data.csv')
hmm_result['fixations'].to_csv('fixations_hmm.csv', index=False)
hmm_result['saccades'].to_csv('saccades_hmm.csv', index=False)

# 5. Визуализация
plotter = EyeMovementPlotter()
plotter.plot_raw_data('pupil_data.csv', 'plots/raw.html')
plotter.plot_with_segmentation('pupil_data.csv', 'fixations_ivt.csv', 
                               'saccades_ivt.csv', 'plots/segmented.html')
```

### Обработка изображений

```python
from src.detection import PupilDetector

detector = PupilDetector()
detector.process_image_folder('data/images/', output_csv='pupil_data.csv')
```

### Использование с компенсацией движения головы

```python
from src.detection import PupilHeadDetector

detector = PupilHeadDetector(head_movement_threshold=0.01)
detector.process_video('video.mp4', output_csv='pupil_data.csv')
# В результате будут сохранены относительные координаты
```

## Алгоритмы сегментации

### I-VT (I-Velocity-Threshold)

**Принцип:** Классификация на основе скорости движения глаз.

**Преимущества:**
- Простота реализации
- Быстрая обработка
- Хорошо работает с высокочастотными данными

**Параметры:**
- `velocity_threshold` - порог скорости (по умолчанию 110 пикс/с)
- `min_fixation_duration` - минимальная длительность фиксации

**Когда использовать:** Для данных с высокой частотой кадров (>30 FPS)

### I-DT (I-Dispersion-Threshold)

**Принцип:** Классификация на основе пространственной дисперсии точек взгляда.

**Преимущества:**
- Устойчив к шуму в данных
- Не требует вычисления производных
- Хорошо работает при низкой частоте кадров

**Параметры:**
- `dispersion_threshold` - порог дисперсии (по умолчанию 7 пикселей)
- `window_duration` - размер окна анализа (по умолчанию 0.2 с)

**Когда использовать:** Когда данные содержат много шума или низкая частота кадров

### HMM (Hidden Markov Model)

**Принцип:** Статистическое моделирование с помощью скрытых марковских цепей.

**Преимущества:**
- Адаптируется к индивидуальным паттернам движений глаз
- Использует контекст для классификации
- Более точная сегментация

**Параметры:**
- `n_components` - количество состояний (обычно 2)
- `n_iter` - количество итераций обучения

**Когда использовать:** Для более точной сегментации, когда важна точность

## Конфигурация параметров

### Рекомендуемые параметры для разных сценариев

#### Для видео низкого качества
```python
# I-VT с повышенным порогом
segmenter = IVTSegmenter(velocity_threshold=150, min_fixation_duration=0.15)

# I-DT с большим окном
segmenter = IDTSegmenter(dispersion_threshold=10, window_duration=0.3)
```

#### Для видео высокого качества
```python
# I-VT с низким порогом
segmenter = IVTSegmenter(velocity_threshold=80, min_fixation_duration=0.05)

# I-DT с малым окном
segmenter = IDTSegmenter(dispersion_threshold=5, window_duration=0.1)
```

#### Для компенсации движения головы
```python
# Уменьшить порог движения головы
detector = PupilHeadDetector(head_movement_threshold=0.005)

# Увеличить порог для игнорирования мелких движений
detector = PupilHeadDetector(head_movement_threshold=0.02)
```

## Дополнительные возможности

### Обработка данных из других источников

Если у вас есть данные в другом формате, можно легко адаптировать:

```python
import pandas as pd

# Загрузка данных
df = pd.read_csv('custom_data.csv')

# Преобразование формата
pupil_data = []
for _, row in df.iterrows():
    pupil_data.append({
        'filename': row['timestamp'],
        'left_x': row['pupil_left_x'],
        'left_y': row['pupil_left_y'],
        'right_x': row['pupil_right_x'],
        'right_y': row['pupil_right_y']
    })

# Сохранение в нужном формате
pd.DataFrame(pupil_data).to_csv('pupil_data.csv', index=False)
```

### Экспорт результатов в Excel

```python
import pandas as pd

# Загрузка результатов
fixations = pd.read_csv('fixations.csv')
saccades = pd.read_csv('saccades.csv')

# Экспорт в Excel
with pd.ExcelWriter('results.xlsx') as writer:
    fixations.to_excel(writer, sheet_name='Fixations', index=False)
    saccades.to_excel(writer, sheet_name='Saccades', index=False)
```

## Решение проблем

### Ошибка: "Cannot open video"
- Проверьте правильность пути к файлу
- Убедитесь, что формат видео поддерживается OpenCV
- Попробуйте использовать полный путь к файлу

### Низкая точность детектирования
- Убедитесь, что лицо хорошо освещено
- Проверьте, что лицо полностью видно в кадре
- Попробуйте увеличить разрешение видео

### Слишком много/мало фиксаций
- Отрегулируйте пороги (`velocity_threshold` или `dispersion_threshold`)
- Измените минимальную длительность фиксации (`min_fixation_duration`)

## Лицензия

-

## Контакты

-

## Благодарности

Проект использует следующие библиотеки:
- MediaPipe для детектирования лица
- OpenCV для обработки изображений
- Pandas для работы с данными
- Plotly для визуализации
- SciPy для обработки сигналов
- HMMlearn для HMM сегментации

