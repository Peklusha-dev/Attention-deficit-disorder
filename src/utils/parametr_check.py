"""
Скрипт для подбора оптимальных параметров алгоритмов сегментации
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.segmentation import IVTSegmenter, IDTSegmenter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def parameter_sweep_analysis(pupil_data_path: str, output_dir: str = "parameter_tuning"):
    """
    Анализ влияния параметров на результаты сегментации
    """

    # Загрузка данных
    pupil_data = pd.read_csv(pupil_data_path)

    # Диапазоны параметров для перебора
    ivt_thresholds = [80, 90, 100, 110, 120, 130, 140, 150]
    idt_thresholds = [3, 4, 5, 6, 7, 8, 9, 10]
    min_durations = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2]

    results = []

    print("Анализ параметров I-VT алгоритма:")
    print("=" * 50)

    for threshold in ivt_thresholds:
        for min_dur in min_durations:
            try:
                segmenter = IVTSegmenter(
                    velocity_threshold=threshold,
                    min_fixation_duration=min_dur,
                    fps=30
                )
                results_seg = segmenter.segment(pupil_data_path)

                fixations = results_seg['fixations']
                saccades = results_seg['saccades']

                # Статистика
                total_fixation_time = sum(fixations['end_time'] - fixations['start_time']) if len(fixations) > 0 else 0
                avg_fixation_duration = total_fixation_time / len(fixations) if len(fixations) > 0 else 0

                results.append({
                    'algorithm': 'IVT',
                    'velocity_threshold': threshold,
                    'min_fixation_duration': min_dur,
                    'n_fixations': len(fixations),
                    'n_saccades': len(saccades),
                    'total_fixation_time': total_fixation_time,
                    'avg_fixation_duration': avg_fixation_duration,
                    'fixation_ratio': len(fixations) / (len(fixations) + len(saccades)) if (len(fixations) + len(saccades)) > 0 else 0
                })

                print(f"IVT threshold={threshold}, min_dur={min_dur}: "
                      f"фиксаций={len(fixations)}, саккад={len(saccades)}, "
                      f"ср.длит.фикс={avg_fixation_duration:.3f}с")

            except Exception as e:
                print(f"Ошибка для IVT threshold={threshold}, min_dur={min_dur}: {e}")

    print("\nАнализ параметров I-DT алгоритма:")
    print("=" * 50)

    for threshold in idt_thresholds:
        for min_dur in min_durations:
            try:
                segmenter = IDTSegmenter(
                    dispersion_threshold=threshold,
                    min_fixation_duration=min_dur,
                    fps=30
                )
                results_seg = segmenter.segment(pupil_data_path)

                fixations = results_seg['fixations']
                saccades = results_seg['saccades']

                total_fixation_time = sum(fixations['end_time'] - fixations['start_time']) if len(fixations) > 0 else 0
                avg_fixation_duration = total_fixation_time / len(fixations) if len(fixations) > 0 else 0

                results.append({
                    'algorithm': 'IDT',
                    'dispersion_threshold': threshold,
                    'min_fixation_duration': min_dur,
                    'n_fixations': len(fixations),
                    'n_saccades': len(saccades),
                    'total_fixation_time': total_fixation_time,
                    'avg_fixation_duration': avg_fixation_duration,
                    'fixation_ratio': len(fixations) / (len(fixations) + len(saccades)) if (len(fixations) + len(saccades)) > 0 else 0
                })

                print(f"IDT threshold={threshold}, min_dur={min_dur}: "
                      f"фиксаций={len(fixations)}, саккад={len(saccades)}, "
                      f"ср.длит.фикс={avg_fixation_duration:.3f}с")

            except Exception as e:
                print(f"Ошибка для IDT threshold={threshold}, min_dur={min_dur}: {e}")

    # Сохранение результатов
    results_df = pd.DataFrame(results)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    results_df.to_csv(output_dir / "parameter_sweep_results.csv", index=False)

    # Визуализация результатов
    create_parameter_plots(results_df, output_dir)

    return results_df

def create_parameter_plots(results_df: pd.DataFrame, output_dir: Path):
    """Создание графиков для анализа параметров"""

    # Фильтрация результатов по алгоритмам
    ivt_results = results_df[results_df['algorithm'] == 'IVT']
    idt_results = results_df[results_df['algorithm'] == 'IDT']

    # График для IVT
    if len(ivt_results) > 0:
        fig_ivt = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Количество фиксаций vs Порог скорости",
                "Средняя длительность фиксаций vs Порог скорости",
                "Общее время фиксаций vs Порог скорости",
                "Доля фиксаций vs Порог скорости"
            )
        )

        colors = ivt_results['min_fixation_duration'].unique()
        color_map = {dur: f'hsl({i*30}, 70%, 50%)' for i, dur in enumerate(colors)}

        for min_dur in colors:
            subset = ivt_results[ivt_results['min_fixation_duration'] == min_dur]

            fig_ivt.add_trace(
                go.Scatter(x=subset['velocity_threshold'], y=subset['n_fixations'],
                          name=f'min_dur={min_dur}', mode='lines+markers',
                          line=dict(color=color_map[min_dur])),
                row=1, col=1
            )

            fig_ivt.add_trace(
                go.Scatter(x=subset['velocity_threshold'], y=subset['avg_fixation_duration'],
                          name=f'min_dur={min_dur}', mode='lines+markers',
                          line=dict(color=color_map[min_dur]), showlegend=False),
                row=1, col=2
            )

            fig_ivt.add_trace(
                go.Scatter(x=subset['velocity_threshold'], y=subset['total_fixation_time'],
                          name=f'min_dur={min_dur}', mode='lines+markers',
                          line=dict(color=color_map[min_dur]), showlegend=False),
                row=2, col=1
            )

            fig_ivt.add_trace(
                go.Scatter(x=subset['velocity_threshold'], y=subset['fixation_ratio'],
                          name=f'min_dur={min_dur}', mode='lines+markers',
                          line=dict(color=color_map[min_dur]), showlegend=False),
                row=2, col=2
            )

        fig_ivt.update_layout(height=800, title_text="Анализ параметров I-VT алгоритма")
        fig_ivt.write_html(output_dir / "ivt_parameter_analysis.html")

    # График для IDT
    if len(idt_results) > 0:
        fig_idt = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Количество фиксаций vs Порог дисперсии",
                "Средняя длительность фиксаций vs Порог дисперсии",
                "Общее время фиксаций vs Порог дисперсии",
                "Доля фиксаций vs Порог дисперсии"
            )
        )

        colors = idt_results['min_fixation_duration'].unique()
        color_map = {dur: f'hsl({i*30}, 70%, 50%)' for i, dur in enumerate(colors)}

        for min_dur in colors:
            subset = idt_results[idt_results['min_fixation_duration'] == min_dur]

            fig_idt.add_trace(
                go.Scatter(x=subset['dispersion_threshold'], y=subset['n_fixations'],
                          name=f'min_dur={min_dur}', mode='lines+markers',
                          line=dict(color=color_map[min_dur])),
                row=1, col=1
            )

            fig_idt.add_trace(
                go.Scatter(x=subset['dispersion_threshold'], y=subset['avg_fixation_duration'],
                          name=f'min_dur={min_dur}', mode='lines+markers',
                          line=dict(color=color_map[min_dur]), showlegend=False),
                row=1, col=2
            )

            fig_idt.add_trace(
                go.Scatter(x=subset['dispersion_threshold'], y=subset['total_fixation_time'],
                          name=f'min_dur={min_dur}', mode='lines+markers',
                          line=dict(color=color_map[min_dur]), showlegend=False),
                row=2, col=1
            )

            fig_idt.add_trace(
                go.Scatter(x=subset['dispersion_threshold'], y=subset['fixation_ratio'],
                          name=f'min_dur={min_dur}', mode='lines+markers',
                          line=dict(color=color_map[min_dur]), showlegend=False),
                row=2, col=2
            )

        fig_idt.update_layout(height=800, title_text="Анализ параметров I-DT алгоритма")
        fig_idt.write_html(output_dir / "idt_parameter_analysis.html")

def quick_parameter_check(pupil_data_path: str):
    """
    Быстрая проверка параметров с визуализацией
    """
    from src.segmentation import IVTSegmenter, IDTSegmenter

    # Тестовые параметры
    test_params = [
        # IVT параметры
        {'alg': 'IVT', 'velocity_threshold': 80, 'min_dur': 0.1, 'name': 'IVT-низкий_порог'},
        {'alg': 'IVT', 'velocity_threshold': 110, 'min_dur': 0.1, 'name': 'IVT-средний_порог'},
        {'alg': 'IVT', 'velocity_threshold': 140, 'min_dur': 0.1, 'name': 'IVT-высокий_порог'},

        # IDT параметры
        {'alg': 'IDT', 'dispersion_threshold': 4, 'min_dur': 0.1, 'name': 'IDT-низкий_порог'},
        {'alg': 'IDT', 'dispersion_threshold': 7, 'min_dur': 0.1, 'name': 'IDT-средний_порог'},
        {'alg': 'IDT', 'dispersion_threshold': 10, 'min_dur': 0.1, 'name': 'IDT-высокий_порог'},
    ]

    pupil_data = pd.read_csv(pupil_data_path)

    fig = make_subplots(
        rows=len(test_params), cols=2,
        subplot_titles=[f"{p['name']} - X" for p in test_params] +
                      [f"{p['name']} - Y" for p in test_params],
        shared_xaxes=True,
        vertical_spacing=0.02
    )

    for i, params in enumerate(test_params):
        try:
            if params['alg'] == 'IVT':
                segmenter = IVTSegmenter(
                    velocity_threshold=params['velocity_threshold'],
                    min_fixation_duration=params['min_dur']
                )
            else:
                segmenter = IDTSegmenter(
                    dispersion_threshold=params['dispersion_threshold'],
                    min_fixation_duration=params['min_dur']
                )

            results = segmenter.segment(pupil_data_path)

            # Визуализация
            plot_data = pupil_data.copy()
            plot_data['time'] = plot_data['filename'].apply(lambda x: parse_time(x) if 'time' not in plot_data.columns else plot_data['time'])
            plot_data['x'] = (plot_data['left_x'] + plot_data['right_x']) / 2
            plot_data['y'] = (plot_data['left_y'] + plot_data['right_y']) / 2
            plot_data['movement_type'] = 'unknown'

            for _, fix in results['fixations'].iterrows():
                mask = (plot_data['time'] >= fix['start_time']) & (plot_data['time'] <= fix['end_time'])
                plot_data.loc[mask, 'movement_type'] = 'fixation'

            for _, sac in results['saccades'].iterrows():
                mask = (plot_data['time'] >= sac['start_time']) & (plot_data['time'] <= sac['end_time'])
                plot_data.loc[mask, 'movement_type'] = 'saccade'

            fixation_points = plot_data[plot_data['movement_type'] == 'fixation']
            saccade_points = plot_data[plot_data['movement_type'] == 'saccade']
            unknown_points = plot_data[plot_data['movement_type'] == 'unknown']

            row = i + 1

            # X координата
            fig.add_trace(
                go.Scatter(x=plot_data['time'], y=plot_data['x'],
                          mode='lines', name='траектория',
                          line=dict(color='lightgray', width=1),
                          showlegend=(i==0)),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(x=fixation_points['time'], y=fixation_points['x'],
                          mode='markers', name='фиксации',
                          marker=dict(color='green', size=4),
                          showlegend=(i==0)),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(x=saccade_points['time'], y=saccade_points['x'],
                          mode='markers', name='саккады',
                          marker=dict(color='red', size=4),
                          showlegend=(i==0)),
                row=row, col=1
            )

            # Y координата
            fig.add_trace(
                go.Scatter(x=plot_data['time'], y=plot_data['y'],
                          mode='lines', name='траектория',
                          line=dict(color='lightgray', width=1),
                          showlegend=False),
                row=row, col=2
            )
            fig.add_trace(
                go.Scatter(x=fixation_points['time'], y=fixation_points['y'],
                          mode='markers', name='фиксации',
                          marker=dict(color='green', size=4),
                          showlegend=False),
                row=row, col=2
            )
            fig.add_trace(
                go.Scatter(x=saccade_points['time'], y=saccade_points['y'],
                          mode='markers', name='саккады',
                          marker=dict(color='red', size=4),
                          showlegend=False),
                row=row, col=2
            )

            print(f"{params['name']}: фиксаций={len(results['fixations'])}, саккад={len(results['saccades'])}")

        except Exception as e:
            print(f"Ошибка для {params['name']}: {e}")

    fig.update_layout(height=300*len(test_params), title_text="Быстрая проверка параметров")
    fig.write_html("quick_parameter_check.html")

# Функция parse_time (добавьте если нет)
def parse_time(filename: str) -> float:
    try:
        parts = filename.replace('frame_', '').replace('.jpg', '').split('_')
        if len(parts) >= 3:
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            microseconds = int(parts[3]) if len(parts) > 3 else 0
            return seconds + microseconds / 1_000_000
        return 0.0
    except:
        return 0.0

if __name__ == '__main__':
    # Быстрая проверка
    quick_parameter_check("моргания.csv")

    # Полный анализ (занимает больше времени)
    # results = parameter_sweep_analysis("моргания.csv")