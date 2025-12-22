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
8. [Валидация и разметка](#валидация-и-разметка)
9. [Конфигурация параметров](#конфигурация-параметров)

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
- **matplotlib** >= 3.4.0 - построение графиков
- **dash** ~= 3.3.0 - интерактивные веб-приложения

## Структура проекта

```
Attention-deficit-disorder/
├── src/                      # Основной код проекта
│   ├── detection/            # Детектирование зрачков
│   │   ├── __init__.py
│   │   └── pupil_detector.py
│   ├── segmentation/        # Алгоритмы сегментации
│   │   ├── __init__.py
│   │   ├── ivt.py           # I-VT алгоритм
│   │   ├── idt.py           # I-DT алгоритм
│   │   └── hmm.py           # HMM алгоритм
│   ├── visualization/        # Визуализация
│   │   ├── __init__.py
│   │   └── plotter.py
│   └── utils/               # Утилиты
│       ├── __init__.py
│       ├── validate_detection.py
│       ├── video_to_frames.py
│       └── rename_img_on_dir.py
├── examples/                 # Примеры использования
│   └── main_example.py
├── markup_utils/             # Инструменты для разметки данных
│   ├── interactive_trajectory_marker.py
│   ├── quick_markup.py
│   └── eyes_tasks.html
├── dataset/                  # Данные и результаты
│   ├── markup/              # Разметка данных
│   └── treking_results/      # Результаты отслеживания
├── test_data/                # Тестовые данные
│   ├── foto/                # Тестовые изображения
│   └── video/                # Тестовые видео
├── archive/                  # Архивные файлы и старые версии
├── requirements.txt          # Зависимости
├── setup.py                 # Установка пакета
└── README.md                # Эта документация
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

### Пример 2: Обработка папки с изображениями

```python
from src.detection import PupilDetector

detector = PupilDetector()
detector.process_image_folder('test_data/foto/blizko', output_csv='pupil_data.csv')
```

### Пример 3: Сегментация движений глаз

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

### Пример 4: Визуализация

```python
from src.visualization import EyeMovementPlotter

# Создание построителя графиков
plotter = EyeMovementPlotter()

# Построение графика сырых данных
plotter.plot_raw_data('pupil_data.csv', 'raw_data.html')

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
- **Возвращает:** Словарь с координатами 'left' и 'right' (x, y) в нормализованном формате [0, 1]

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

##### `calculate_total_error(csv_file, image_folder, verbose=False)`
- **Описание:** Вычисляет среднюю ошибку детектирования для всех изображений из CSV файла разметки
- **Параметры:**
  - `csv_file` (str): Путь к CSV файлу с разметкой (ground truth)
  - `image_folder` (str): Путь к папке с изображениями
  - `verbose` (bool): Выводить ли информацию о каждом кадре
- **Возвращает:** Средняя ошибка в пикселях

##### `process_image_folder_with_validation(input_folder, annotations_csv, output_csv, output_errors_csv)`
- **Описание:** Обрабатывает изображения с валидацией против разметки и сохранением ошибок
- **Параметры:**
  - `input_folder` (str): Путь к папке с изображениями
  - `annotations_csv` (str): Путь к CSV файлу с разметкой
  - `output_csv` (str): Путь для сохранения результатов детектирования
  - `output_errors_csv` (str): Путь для сохранения ошибок валидации

### PupilHeadDetector (src/detection/pupil_detector.py)

**Назначение:** Детектор зрачков с компенсацией движения головы.

**Наследуется от:** PupilDetector

**Дополнительные возможности:**
- Автоматическая компенсация движения головы
- Использует референсные точки лица (переносица, уголки глаз)
- Сохраняет относительные координаты зрачков (`rel_left_x`, `rel_left_y`, `rel_right_x`, `rel_right_y`)

**Конструктор:**
```python
PupilHeadDetector(head_movement_threshold=0.01)
```

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
  - `output_prefix` (str): Префикс для выходных файлов (не используется)
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
filename,left_x,left_y,right_x,right_y,left_x_norm,left_y_norm,right_x_norm,right_y_norm
frame_0_00_00_000000.jpg,320,240,480,240,0.5,0.5,0.75,0.5
frame_0_00_00_033333.jpg,325,242,485,242,0.508,0.504,0.758,0.504
...
```

**Структура:**
- `filename` - имя файла/идентификатор кадра
- `left_x`, `left_y` - координаты левого зрачка в пикселях
- `right_x`, `right_y` - координаты правого зрачка в пикселях
- `left_x_norm`, `left_y_norm` - нормализованные координаты левого зрачка [0, 1]
- `right_x_norm`, `right_y_norm` - нормализованные координаты правого зрачка [0, 1]

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

### Формат разметки (ground truth)

```csv
filename,left_x,left_y,right_x,right_y
frame_0_00_00_000000.jpg,320,240,480,240
...
```

## Примеры использования

### Полный pipeline

```python
from src.detection import PupilDetector
from src.segmentation import IVTSegmenter, IDTSegmenter, HMMSegmenter
from src.visualization import EyeMovementPlotter

# 1. Детектирование зрачков
detector = PupilDetector()
detector.process_video('test_data/video_7.MOV', output_csv='pupil_data.csv')

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
detector.process_image_folder('test_data/foto/blizko', output_csv='pupil_data.csv')
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

## Валидация и разметка

### Валидация детектирования

Проект включает инструменты для валидации точности детектирования зрачков:

```python
from src.detection import PupilDetector

detector = PupilDetector()

# Вычисление средней ошибки для всех изображений из разметки
average_error = detector.calculate_total_error(
    csv_file='dataset/markup/test_7.csv',
    image_folder='test_data/video/video7_frames',
    verbose=False
)

# Обработка с валидацией и сохранением ошибок
detector.process_image_folder_with_validation(
    input_folder='test_data/video/video7_frames',
    annotations_csv='dataset/markup/test_7.csv',
    output_csv='pupil_data_7.csv',
    output_errors_csv='dataset/treking_results/validation_results7.csv'
)
```

### Инструменты разметки

В папке `markup_utils/` находятся инструменты для создания разметки данных:
- `interactive_trajectory_marker.py` - интерактивная разметка траекторий
- `quick_markup.py` - быстрая разметка координат зрачков
- `eyes_tasks.html` - веб-интерфейс для проведения глазодвигательных экспериментов

### Интерактивное приложение разметки траекторий

**Файл:** `markup_utils/interactive_trajectory_marker.py`

Веб-приложение на базе Dash для интерактивной разметки траекторий движений зрачков. Позволяет визуально выделять диапазоны времени на графике и помечать их как саккады или фиксации.

#### Возможности:
- Визуализация траектории движений зрачков (горизонтальное и вертикальное движение)
- Интерактивное выделение диапазонов времени на графике
- Классификация выделенных диапазонов как саккады или фиксации
- Сохранение разметки в CSV файл
- Загрузка и продолжение работы с существующей разметкой

#### Использование:

1. **Настройка параметров** в начале файла:
```python
VIDEO_PATH = None  # или путь к видео
IMAGE_FOLDER = "../test_data/video/video7_frames"  # путь к папке с изображениями
OUTPUT_PUPIL_CSV = "../pupil_data_trajectory7.csv"
OUTPUT_ANNOTATIONS_CSV = "../trajectory_annotations7.csv"
DETECTOR_TYPE = "basic"  # "basic" или "head_compensated"
```

2. **Запуск приложения:**
```bash
python markup_utils/interactive_trajectory_marker.py
```

3. **Открытие в браузере:**
   - Приложение автоматически откроется по адресу `http://127.0.0.1:8053`
   - Если данные о зрачках еще не обработаны, они будут обработаны автоматически

4. **Работа с интерфейсом:**
   - Используйте инструмент выделения на панели инструментов графика (иконка прямоугольника или лассо)
   - Выделите диапазон времени на графике
   - Выберите тип события: "Саккада" или "Фиксация"
   - Нажмите "Сохранить выделенный диапазон"
   - Повторите для всех нужных диапазонов
   - Нажмите "Сохранить все в CSV" для сохранения разметки

5. **Результат:**
   - CSV файл с колонками: `type`, `start_time`, `end_time`
   - Разметка отображается на графике цветными областями (красный - саккады, зеленый - фиксации)

![Скриншот приложения разметки траекторий](screenshots/trajectory_marker.png)
*Интерфейс интерактивной разметки траекторий движений зрачков*

### Быстрая разметка координат зрачков

**Файл:** `markup_utils/quick_markup.py`

OpenCV приложение для быстрой ручной разметки координат зрачков на изображениях. Позволяет кликать по зрачкам на каждом изображении и сохранять их координаты.

#### Возможности:
- Просмотр изображений по одному
- Клик по двум точкам (левый и правый зрачок)
- Выбор типа события (моргание, саккада, фиксация)
- Масштабирование и перемещение изображения
- Сохранение координат в CSV

#### Использование:

1. **Настройка параметров** в начале файла:
```python
image_folder = '../test_data/video/video7_frames'  # папка с изображениями
output_csv = 'annotations_7.csv'  # файл для сохранения
screen_width = 1080  # размер экрана
screen_height = 600
```

2. **Запуск:**
```bash
python markup_utils/quick_markup.py
```

3. **Управление:**
   - **ЛКМ** - клик по зрачкам (2 клика: левый и правый)
   - **ENTER** - подтвердить разметку и перейти к следующему изображению
   - **BACKSPACE** - удалить последнюю точку
   - **ESC** - выйти и сохранить
   - **M** - выбрать тип события: Моргание (Blink)
   - **S** - выбрать тип события: Саккада (Saccade)
   - **F** - выбрать тип события: Фиксация (Fixation)
   - **X** - очистить тип события
   - **+/-** - масштабирование
   - **Ctrl + колесо мыши** - масштабирование
   - **Перетаскивание мышью** - перемещение изображения
   - **A/D** или стрелки - переход к предыдущему/следующему изображению
   - **P** - пропустить изображение

4. **Результат:**
   - CSV файл с колонками: `filename`, `left_x`, `left_y`, `right_x`, `right_y`, `event_type`

![Скриншот быстрой разметки](screenshots/quick_markup.png)
*Интерфейс быстрой разметки координат зрачков*

### Глазодвигательные задачи (eyes_tasks)

**Файл:** `markup_utils/eyes_tasks.html`

Веб-приложение для проведения экспериментов с различными глазодвигательными задачами. Генерирует события с временными метками для синхронизации с записью видео.

#### Доступные задачи:

1. **Детерминированные саккады**
   - Точки появляются в строго заданной последовательности
   - Эталон: Фиксация (2 сек) → Саккада → Фиксация (2 сек)

2. **Предсказуемые саккады (горизонтальные)**
   - Точка движется по горизонтали слева направо и обратно
   - Эталон: Регулярные саккады с постоянным интервалом

3. **Плавное слежение**
   - Синий шар движется по круговой траектории
   - Эталон: Непрерывное плавное слежение

4. **Поиск цели**
   - Сетка 10x10 с буквами, нужно найти красный квадрат с буквой T
   - Эталон: Сканирующие саккады → Фиксация на цели

5. **Свободный просмотр изображения**
   - Показывается случайное изображение для свободного рассматривания
   - Эталон: Естественные саккады и фиксации

6. **Чтение текста**
   - Текст для естественного чтения
   - Эталон: Регулярные саккады вперед → Фиксации → Регрессии

#### Использование:

1. **Открытие приложения:**
   - Откройте файл `markup_utils/eyes_tasks.html` в браузере
   - Или разместите на веб-сервере для доступа по сети

2. **Проведение эксперимента:**
   - Выберите задачу из списка
   - **ВАЖНО:** Запустите запись видео **ПЕРЕД** началом задачи
   - Выполните задачу, следуя инструкциям на экране
   - Нажмите "Завершить" по окончании задачи
   - Все события автоматически логируются с временными метками

3. **Просмотр и сохранение лога:**
   - Лог отображается внизу главного экрана
   - Нажмите "Скачать лог" для сохранения JSON файла с событиями
   - Лог содержит:
     - Временные метки в формате ISO
     - Тип события (STIMULUS, FIXATION_START, SACCADE_START, и т.д.)
     - Данные события (координаты, позиции, и т.д.)

4. **Синхронизация с видео:**
   - Используйте временные метки из лога для синхронизации с записью видео
   - События логируются с реальным временем (Date.now())
   - Формат: `{"timestamp": 1234567890, "isoTime": "2024-01-01T12:00:00.000Z", "type": "...", "message": "...", "data": {...}}`

5. **Типы событий в логе:**
   - `SYSTEM` - системные события (инициализация, информация об экране)
   - `EXPERIMENT_START` - начало задачи
   - `EXPERIMENT_END` - завершение задачи
   - `STIMULUS` - появление стимула
   - `FIXATION_START` - начало фиксации
   - `SACCADE_START` - начало саккады
   - `SMOOTH_PURSUIT` - позиция цели при плавном слежении
   - `RESPONSE` - реакция испытуемого (например, нажатие на цель)

![Скриншот глазодвигательных задач](screenshots/eyes_tasks.png)
*Интерфейс глазодвигательных задач*

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

### Проблемы с валидацией
- Убедитесь, что имена файлов в разметке совпадают с именами файлов изображений
- Проверьте формат CSV файла разметки (должен содержать колонки: filename, left_x, left_y, right_x, right_y)

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
