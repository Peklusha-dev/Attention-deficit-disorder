"""
Скрипт для валидации алгоритмов сегментации движений глаз
Сравнивает результаты алгоритмов сегментации (IVT, IDT, HMM) с разметкой (ground truth).
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# =============================================================================
# ПАРАМЕТРЫ КОНФИГУРАЦИИ (изменяйте здесь)
# =============================================================================

# Пути к файлам
PUPIL_DATA_PATH = "../../маятник.csv"
ANNOTATIONS_PATH = "../../маятник_annotations.csv"
OUTPUT_PATH = "segmentation_validation_results.csv"
OUTPUT_DIR = None  # None для автоматического создания директории

# Параметры алгоритмов
IVT_THRESHOLD = 110
IDT_THRESHOLD = 7
FPS = 30
MIN_FIXATION_DURATION = 0.1
IOU_THRESHOLD = 0.1

# Настройки визуализации
CREATE_PLOTS = True

# =============================================================================
# КОНЕЦ ПАРАМЕТРОВ КОНФИГУРАЦИИ
# =============================================================================

# Добавление пути к src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.segmentation import IVTSegmenter, IDTSegmenter, HMMSegmenter
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_time(filename: str) -> float:
    """Парсинг времени из имени файла в формате 'frame_0_00_00_000000.jpg'."""
    try:
        parts = filename.replace('frame_', '').replace('.jpg', '').split('_')
        if len(parts) >= 3:
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            microseconds = int(parts[3]) if len(parts) > 3 else 0
            return seconds + microseconds / 1_000_000
        return 0.0
    except:
        return 0.0


def load_annotations(annotations_path: str) -> pd.DataFrame:
    """Загрузка разметки из CSV файла."""
    df = pd.read_csv(annotations_path)

    # Проверка наличия необходимых колонок
    required_cols = ['type', 'start_time', 'end_time']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Файл разметки должен содержать колонки: {required_cols}")

    # Нормализация типов событий
    df['type'] = df['type'].str.lower().str.strip()

    # Сортировка по времени начала
    df = df.sort_values('start_time').reset_index(drop=True)

    return df


def get_time_range(pupil_data: pd.DataFrame) -> Tuple[float, float]:
    """Получение диапазона времени из данных о зрачках."""
    if 'time' not in pupil_data.columns:
        pupil_data['time'] = pupil_data['filename'].apply(parse_time)

    return pupil_data['time'].min(), pupil_data['time'].max()


def filter_annotations_by_time_range(annotations: pd.DataFrame,
                                   min_time: float,
                                   max_time: float) -> pd.DataFrame:
    """Фильтрация разметки по диапазону времени данных."""
    # Фильтрация событий, которые хотя бы частично попадают в диапазон
    mask = (annotations['end_time'] >= min_time) & (annotations['start_time'] <= max_time)
    filtered = annotations[mask].copy()

    # Обрезка событий до границ диапазона
    filtered['start_time'] = filtered['start_time'].clip(lower=min_time)
    filtered['end_time'] = filtered['end_time'].clip(upper=max_time)

    # Пересчет длительности и удаление нулевых событий
    filtered['duration'] = filtered['end_time'] - filtered['start_time']
    filtered = filtered[filtered['duration'] > 0].reset_index(drop=True)

    return filtered


def calculate_overlap(interval1: Tuple[float, float], interval2: Tuple[float, float]) -> float:
    """Вычисление перекрытия между двумя временными интервалами."""
    start1, end1 = interval1
    start2, end2 = interval2

    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    return max(0, overlap_end - overlap_start)


def calculate_iou(interval1: Tuple[float, float], interval2: Tuple[float, float]) -> float:
    """Вычисление Intersection over Union (IoU) для двух временных интервалов."""
    start1, end1 = interval1
    start2, end2 = interval2

    intersection = calculate_overlap(interval1, interval2)
    union = (end1 - start1) + (end2 - start2) - intersection

    return intersection / union if union > 0 else 0.0


def match_events(predicted: pd.DataFrame, ground_truth: pd.DataFrame, event_type: str, iou_threshold: float = 0.1) -> Dict:
    """Сопоставление предсказанных событий с разметкой."""
    # Фильтрация по типу события
    gt_filtered = ground_truth[ground_truth['type'] == event_type].copy()

    # Обработка краевых случаев
    if len(predicted) == 0 and len(gt_filtered) == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'mean_iou': 1.0}

    if len(predicted) == 0:
        return {'tp': 0, 'fp': 0, 'fn': len(gt_filtered), 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mean_iou': 0.0}

    if len(gt_filtered) == 0:
        return {'tp': 0, 'fp': len(predicted), 'fn': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mean_iou': 0.0}

    # Создание матрицы IoU
    iou_matrix = np.zeros((len(predicted), len(gt_filtered)))

    for i, (_, pred_row) in enumerate(predicted.iterrows()):
        pred_interval = (pred_row['start_time'], pred_row['end_time'])
        for j, (_, gt_row) in enumerate(gt_filtered.iterrows()):
            gt_interval = (gt_row['start_time'], gt_row['end_time'])
            iou_matrix[i, j] = calculate_iou(pred_interval, gt_interval)

    # Жадное сопоставление
    matched_pred = set()
    matched_gt = set()
    tp = 0
    ious = []

    matches = []
    for i in range(len(predicted)):
        for j in range(len(gt_filtered)):
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((i, j, iou_matrix[i, j]))

    matches.sort(key=lambda x: x[2], reverse=True)

    for i, j, iou in matches:
        if i not in matched_pred and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)
            tp += 1
            ious.append(iou)

    fp = len(predicted) - len(matched_pred)
    fn = len(gt_filtered) - len(matched_gt)

    # Вычисление метрик
    precision = tp / len(predicted) if len(predicted) > 0 else 0.0
    recall = tp / len(gt_filtered) if len(gt_filtered) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = np.mean(ious) if ious else 0.0

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1, 'mean_iou': mean_iou
    }


def evaluate_segmentation_algorithm(pupil_data_path: str, annotations_path: str,
                                  algorithm_name: str, segmenter, output_dir: Optional[Path] = None) -> Dict:
    """Оценка одного алгоритма сегментации."""
    print(f"\n{'='*60}")
    print(f"Оценка алгоритма: {algorithm_name}")
    print(f"{'='*60}")

    # Загрузка данных
    pupil_data = pd.read_csv(pupil_data_path)
    annotations = load_annotations(annotations_path)

    # Получение диапазона времени данных
    min_time, max_time = get_time_range(pupil_data)
    print(f"Диапазон времени в данных: {min_time:.3f} - {max_time:.3f} сек")

    # Фильтрация разметки
    filtered_annotations = filter_annotations_by_time_range(annotations, min_time, max_time)
    print(f"Событий в разметке: {len(filtered_annotations)}")

    # Выполнение сегментации
    print(f"Выполнение сегментации...")
    try:
        results = segmenter.segment(pupil_data_path)
        fixations = results['fixations']
        saccades = results['saccades']
    except Exception as e:
        print(f"Ошибка при выполнении сегментации: {e}")
        return {'algorithm': algorithm_name, 'error': str(e)}, None, None, None

    print(f"Предсказано: фиксаций={len(fixations)}, саккад={len(saccades)}")

    # Оценка качества
    fixation_metrics = match_events(fixations, filtered_annotations, 'fixation', IOU_THRESHOLD)
    saccade_metrics = match_events(saccades, filtered_annotations, 'saccade', IOU_THRESHOLD)

    # Объединение результатов
    results_dict = {
        'algorithm': algorithm_name,
        'error': None,

        # Фиксации
        'fixations_precision': fixation_metrics['precision'],
        'fixations_recall': fixation_metrics['recall'],
        'fixations_f1': fixation_metrics['f1'],
        'fixations_mean_iou': fixation_metrics['mean_iou'],
        'fixations_tp': fixation_metrics['tp'],
        'fixations_fp': fixation_metrics['fp'],
        'fixations_fn': fixation_metrics['fn'],

        # Саккады
        'saccades_precision': saccade_metrics['precision'],
        'saccades_recall': saccade_metrics['recall'],
        'saccades_f1': saccade_metrics['f1'],
        'saccades_mean_iou': saccade_metrics['mean_iou'],
        'saccades_tp': saccade_metrics['tp'],
        'saccades_fp': saccade_metrics['fp'],
        'saccades_fn': saccade_metrics['fn'],
    }

    # Вывод результатов
    print(f"Фиксации: Precision={fixation_metrics['precision']:.3f}, Recall={fixation_metrics['recall']:.3f}, F1={fixation_metrics['f1']:.3f}")
    print(f"Саккады: Precision={saccade_metrics['precision']:.3f}, Recall={saccade_metrics['recall']:.3f}, F1={saccade_metrics['f1']:.3f}")

    return results_dict, fixations, saccades, filtered_annotations


def create_interactive_plot(pupil_data: pd.DataFrame, fixations: pd.DataFrame, saccades: pd.DataFrame,
                          annotations: Optional[pd.DataFrame], algorithm_name: str, output_file: str):
    """Создание интерактивного графика с траекторией движения и выделенными событиями."""
    # Подготовка данных
    if 'time' not in pupil_data.columns:
        pupil_data['time'] = pupil_data['filename'].apply(parse_time)

    # Вычисление средней позиции зрачков
    pupil_data['x'] = (pupil_data['left_x'] + pupil_data['right_x']) / 2
    pupil_data['y'] = (pupil_data['left_y'] + pupil_data['right_y']) / 2

    # Инициализация типа движения для каждой точки
    pupil_data['movement_type'] = 'unknown'

    # Отметка точек, относящихся к фиксациям (зеленый)
    if fixations is not None and len(fixations) > 0:
        for _, fix in fixations.iterrows():
            mask = (pupil_data['time'] >= fix['start_time']) & (pupil_data['time'] <= fix['end_time'])
            pupil_data.loc[mask, 'movement_type'] = 'fixation'

    # Отметка точек, относящихся к саккадам (красный)
    if saccades is not None and len(saccades) > 0:
        for _, sac in saccades.iterrows():
            mask = (pupil_data['time'] >= sac['start_time']) & (pupil_data['time'] <= sac['end_time'])
            pupil_data.loc[mask, 'movement_type'] = 'saccade'

    # Разделение данных по типу движения
    fixation_points = pupil_data[pupil_data['movement_type'] == 'fixation']
    saccade_points = pupil_data[pupil_data['movement_type'] == 'saccade']
    unknown_points = pupil_data[pupil_data['movement_type'] == 'unknown']

    # Создание фигуры с двумя подграфиками
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"Траектория движения глаз - {algorithm_name} (X координата)",
            f"Траектория движения глаз - {algorithm_name} (Y координата)"
        ),
        shared_xaxes=True,
        vertical_spacing=0.1
    )

    # Полная траектория (серым)
    fig.add_trace(
        go.Scatter(
            x=pupil_data['time'],
            y=pupil_data['x'],
            mode='lines',
            name='Полная траектория',
            line=dict(color='lightgray', width=1),
            showlegend=True,
            hovertemplate='Время: %{x:.3f}с<br>X: %{y:.1f}px<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=pupil_data['time'],
            y=pupil_data['y'],
            mode='lines',
            name='Полная траектория',
            line=dict(color='lightgray', width=1),
            showlegend=False,
            hovertemplate='Время: %{x:.3f}с<br>Y: %{y:.1f}px<extra></extra>'
        ),
        row=2, col=1
    )

    # Фиксации (зеленые точки)
    if len(fixation_points) > 0:
        fig.add_trace(
            go.Scatter(
                x=fixation_points['time'],
                y=fixation_points['x'],
                mode='markers',
                name='Фиксации',
                marker=dict(color='green', size=6, opacity=0.7),
                showlegend=True,
                hovertemplate='Время: %{x:.3f}с<br>X: %{y:.1f}px<br>Тип: Фиксация<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=fixation_points['time'],
                y=fixation_points['y'],
                mode='markers',
                name='Фиксации',
                marker=dict(color='green', size=6, opacity=0.7),
                showlegend=False,
                hovertemplate='Время: %{x:.3f}с<br>Y: %{y:.1f}px<br>Тип: Фиксация<extra></extra>'
            ),
            row=2, col=1
        )

    # Саккады (красные точки)
    if len(saccade_points) > 0:
        fig.add_trace(
            go.Scatter(
                x=saccade_points['time'],
                y=saccade_points['x'],
                mode='markers',
                name='Саккады',
                marker=dict(color='red', size=6, opacity=0.7),
                showlegend=True,
                hovertemplate='Время: %{x:.3f}с<br>X: %{y:.1f}px<br>Тип: Саккада<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=saccade_points['time'],
                y=saccade_points['y'],
                mode='markers',
                name='Саккады',
                marker=dict(color='red', size=6, opacity=0.7),
                showlegend=False,
                hovertemplate='Время: %{x:.3f}с<br>Y: %{y:.1f}px<br>Тип: Саккада<extra></extra>'
            ),
            row=2, col=1
        )

    # Неопознанные точки (синие точки)
    if len(unknown_points) > 0:
        fig.add_trace(
            go.Scatter(
                x=unknown_points['time'],
                y=unknown_points['x'],
                mode='markers',
                name='Не опознано',
                marker=dict(color='blue', size=4, opacity=0.5),
                showlegend=True,
                hovertemplate='Время: %{x:.3f}с<br>X: %{y:.1f}px<br>Тип: Не опознано<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=unknown_points['time'],
                y=unknown_points['y'],
                mode='markers',
                name='Не опознано',
                marker=dict(color='blue', size=4, opacity=0.5),
                showlegend=False,
                hovertemplate='Время: %{x:.3f}с<br>Y: %{y:.1f}px<br>Тип: Не опознано<extra></extra>'
            ),
            row=2, col=1
        )

    # Добавление разметки (ground truth) в виде затененных областей
    if annotations is not None and len(annotations) > 0:
        for _, ann in annotations.iterrows():
            color = 'lightgreen' if ann['type'] == 'fixation' else 'lightcoral'
            name = 'Разметка: Фиксация' if ann['type'] == 'fixation' else 'Разметка: Саккада'

            # Добавление прямоугольников для разметки
            fig.add_vrect(
                x0=ann['start_time'], x1=ann['end_time'],
                fillcolor=color, opacity=0.2,
                line_width=0,
                annotation_text=name,
                annotation_position="top left",
                row=1, col=1
            )
            fig.add_vrect(
                x0=ann['start_time'], x1=ann['end_time'],
                fillcolor=color, opacity=0.2,
                line_width=0,
                showlegend=False,
                row=2, col=1
            )

    # Настройка осей и layout
    fig.update_xaxes(title_text="Время (секунды)", row=2, col=1)
    fig.update_yaxes(title_text="Позиция X (пиксели)", row=1, col=1)
    fig.update_yaxes(title_text="Позиция Y (пиксели)", row=2, col=1)

    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Сохранение
    fig.write_html(output_file)
    print(f"  График сохранен в: {output_file}")


def main():
    """Основная функция для запуска валидации."""
    # Проверка существования файлов
    pupil_path = Path(PUPIL_DATA_PATH)
    annotations_path = Path(ANNOTATIONS_PATH)

    if not pupil_path.exists():
        print(f"Ошибка: Файл с данными о зрачках не найден: {pupil_path}")
        return 1

    if not annotations_path.exists():
        print(f"Ошибка: Файл с разметкой не найден: {annotations_path}")
        return 1

    print(f"Загрузка данных...")
    print(f"  Данные о зрачках: {pupil_path}")
    print(f"  Разметка: {annotations_path}")

    # Создание сегментаторов
    segmenters = {
        'IVT': IVTSegmenter(
            velocity_threshold=IVT_THRESHOLD,
            min_fixation_duration=MIN_FIXATION_DURATION,
            fps=FPS
        ),
        'IDT': IDTSegmenter(
            dispersion_threshold=IDT_THRESHOLD,
            min_fixation_duration=MIN_FIXATION_DURATION,
            fps=FPS
        ),
        'HMM': HMMSegmenter(
            min_fixation_duration=MIN_FIXATION_DURATION,
            fps=FPS
        )
    }

    # Определение директории для сохранения
    if OUTPUT_DIR:
        output_dir = Path(OUTPUT_DIR)
    else:
        output_path = Path(OUTPUT_PATH)
        output_dir = output_path.parent / f"{output_path.stem}_results"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Результаты сохраняются в: {output_dir}")

    # Загрузка данных о зрачках
    pupil_data = pd.read_csv(pupil_path)

    # Оценка всех алгоритмов
    all_results = []

    for alg_name, segmenter in segmenters.items():
        try:
            result, fixations, saccades, filtered_annotations = evaluate_segmentation_algorithm(
                str(pupil_path),
                str(annotations_path),
                alg_name,
                segmenter,
                output_dir
            )
            all_results.append(result)

            # Создание графика
            if CREATE_PLOTS:
                plot_file = output_dir / f"plot_{alg_name.lower()}.html"
                create_interactive_plot(
                    pupil_data.copy(),
                    fixations,
                    saccades,
                    filtered_annotations,
                    alg_name,
                    str(plot_file)
                )
        except Exception as e:
            print(f"Ошибка при оценке алгоритма {alg_name}: {e}")
            all_results.append({'algorithm': alg_name, 'error': str(e)})

    # Сохранение результатов
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

    print(f"\n{'='*60}")
    print(f"Результаты сохранены в: {OUTPUT_PATH}")
    print(f"{'='*60}")

    # Вывод сводной таблицы
    print(f"\nСводная таблица результатов:")
    successful = results_df[results_df['error'].isna()] if 'error' in results_df.columns else results_df

    if len(successful) > 0:
        summary_cols = ['algorithm', 'fixations_f1', 'fixations_precision', 'fixations_recall',
                      'saccades_f1', 'saccades_precision', 'saccades_recall']
        available_cols = [col for col in summary_cols if col in successful.columns]
        print(successful[available_cols].to_string(index=False))
    else:
        print("Нет успешных результатов для отображения")

    return 0


if __name__ == '__main__':
    sys.exit(main())