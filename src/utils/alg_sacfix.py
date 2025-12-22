"""
Скрипт для запуска 3 алгоритмов сегментации движений глаз и построения интерактивных графиков
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import sys

# =============================================================================
# ПАРАМЕТРЫ КОНФИГУРАЦИИ (изменяйте здесь)
# =============================================================================

# Пути к файлам
PUPIL_DATA_PATH = "../../маятник.csv"
OUTPUT_DIR = "segmentation_results"

# Параметры алгоритмов
IVT_THRESHOLD = 15
IDT_THRESHOLD = 2
FPS = 30
MIN_FIXATION_DURATION = 0.1

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


def run_segmentation_algorithm(pupil_data_path: str, algorithm_name: str, segmenter) -> Dict:
	"""Запуск одного алгоритма сегментации."""
	print(f"\n{'=' * 60}")
	print(f"Запуск алгоритма: {algorithm_name}")
	print(f"{'=' * 60}")

	# Загрузка данных
	pupil_data = pd.read_csv(pupil_data_path)

	# Выполнение сегментации
	print(f"Выполнение сегментации...")
	try:
		results = segmenter.segment(pupil_data_path)
		fixations = results['fixations']
		saccades = results['saccades']

		print(f"Найдено: фиксаций={len(fixations)}, саккад={len(saccades)}")
		return {
			'algorithm': algorithm_name,
			'fixations': fixations,
			'saccades': saccades,
			'error': None
		}

	except Exception as e:
		print(f"Ошибка при выполнении сегментации: {e}")
		return {
			'algorithm': algorithm_name,
			'fixations': pd.DataFrame(),
			'saccades': pd.DataFrame(),
			'error': str(e)
		}


def create_segmentation_plot(pupil_data: pd.DataFrame, fixations: pd.DataFrame, saccades: pd.DataFrame,
                             algorithm_name: str, output_file: str):
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


def create_comparison_plot(pupil_data: pd.DataFrame, all_results: List[Dict], output_file: str):
	"""Создание сравнительного графика всех алгоритмов."""

	# Подготовка данных
	if 'time' not in pupil_data.columns:
		pupil_data['time'] = pupil_data['filename'].apply(parse_time)

	pupil_data['x'] = (pupil_data['left_x'] + pupil_data['right_x']) / 2
	pupil_data['y'] = (pupil_data['left_y'] + pupil_data['right_y']) / 2

	# Создание фигуры с подграфиками для каждого алгоритма
	fig = make_subplots(
		rows=3, cols=2,
		subplot_titles=(
			"IVT - X координата", "IVT - Y координата",
			"IDT - X координата", "IDT - Y координата",
			"HMM - X координата", "HMM - Y координата"
		),
		shared_xaxes=True,
		vertical_spacing=0.05,
		horizontal_spacing=0.05
	)

	colors = {'fixation': 'green', 'saccade': 'red', 'unknown': 'blue'}

	for i, result in enumerate(all_results):
		if result['error'] is not None:
			continue

		algorithm_name = result['algorithm']
		fixations = result['fixations']
		saccades = result['saccades']

		# Создаем копию данных для этого алгоритма
		plot_data = pupil_data.copy()
		plot_data['movement_type'] = 'unknown'

		# Отметка фиксаций
		if len(fixations) > 0:
			for _, fix in fixations.iterrows():
				mask = (plot_data['time'] >= fix['start_time']) & (plot_data['time'] <= fix['end_time'])
				plot_data.loc[mask, 'movement_type'] = 'fixation'

		# Отметка саккад
		if len(saccades) > 0:
			for _, sac in saccades.iterrows():
				mask = (plot_data['time'] >= sac['start_time']) & (plot_data['time'] <= sac['end_time'])
				plot_data.loc[mask, 'movement_type'] = 'saccade'

		# Разделение по типам движения
		fixation_points = plot_data[plot_data['movement_type'] == 'fixation']
		saccade_points = plot_data[plot_data['movement_type'] == 'saccade']
		unknown_points = plot_data[plot_data['movement_type'] == 'unknown']

		row = i + 1

		# Полная траектория
		fig.add_trace(
			go.Scatter(
				x=plot_data['time'], y=plot_data['x'],
				mode='lines', name=f'{algorithm_name} - траектория',
				line=dict(color='lightgray', width=1),
				showlegend=(i == 0),
				hovertemplate='Время: %{x:.3f}с<br>X: %{y:.1f}px<extra></extra>'
			),
			row=row, col=1
		)

		fig.add_trace(
			go.Scatter(
				x=plot_data['time'], y=plot_data['y'],
				mode='lines', name=f'{algorithm_name} - траектория',
				line=dict(color='lightgray', width=1),
				showlegend=False,
				hovertemplate='Время: %{x:.3f}с<br>Y: %{y:.1f}px<extra></extra>'
			),
			row=row, col=2
		)

		# Фиксации
		if len(fixation_points) > 0:
			fig.add_trace(
				go.Scatter(
					x=fixation_points['time'], y=fixation_points['x'],
					mode='markers', name=f'{algorithm_name} - фиксации',
					marker=dict(color='green', size=4, opacity=0.7),
					showlegend=(i == 0),
					hovertemplate='Время: %{x:.3f}с<br>X: %{y:.1f}px<br>Фиксация<extra></extra>'
				),
				row=row, col=1
			)

			fig.add_trace(
				go.Scatter(
					x=fixation_points['time'], y=fixation_points['y'],
					mode='markers', name=f'{algorithm_name} - фиксации',
					marker=dict(color='green', size=4, opacity=0.7),
					showlegend=False,
					hovertemplate='Время: %{x:.3f}с<br>Y: %{y:.1f}px<br>Фиксация<extra></extra>'
				),
				row=row, col=2
			)

		# Саккады
		if len(saccade_points) > 0:
			fig.add_trace(
				go.Scatter(
					x=saccade_points['time'], y=saccade_points['x'],
					mode='markers', name=f'{algorithm_name} - саккады',
					marker=dict(color='red', size=4, opacity=0.7),
					showlegend=(i == 0),
					hovertemplate='Время: %{x:.3f}с<br>X: %{y:.1f}px<br>Саккада<extra></extra>'
				),
				row=row, col=1
			)

			fig.add_trace(
				go.Scatter(
					x=saccade_points['time'], y=saccade_points['y'],
					mode='markers', name=f'{algorithm_name} - саккады',
					marker=dict(color='red', size=4, opacity=0.7),
					showlegend=False,
					hovertemplate='Время: %{x:.3f}с<br>Y: %{y:.1f}px<br>Саккада<extra></extra>'
				),
				row=row, col=2
			)

	# Настройка layout
	fig.update_layout(
		height=1200,
		showlegend=True,
		hovermode='closest',
		title_text="Сравнение алгоритмов сегментации движений глаз"
	)

	# Обновление подписей осей
	for i in range(1, 4):
		fig.update_xaxes(title_text="Время (с)", row=i, col=1)
		fig.update_xaxes(title_text="Время (с)", row=i, col=2)
		fig.update_yaxes(title_text="X (пиксели)", row=i, col=1)
		fig.update_yaxes(title_text="Y (пиксели)", row=i, col=2)

	fig.write_html(output_file)
	print(f"  Сравнительный график сохранен в: {output_file}")


def main():
	"""Основная функция для запуска алгоритмов и построения графиков."""

	# Проверка существования файла
	pupil_path = Path(PUPIL_DATA_PATH)
	if not pupil_path.exists():
		print(f"Ошибка: Файл с данными о зрачках не найден: {pupil_path}")
		return 1

	print(f"Загрузка данных из: {pupil_path}")

	# Создание директории для результатов
	output_dir = Path(OUTPUT_DIR)
	output_dir.mkdir(parents=True, exist_ok=True)
	print(f"Результаты сохраняются в: {output_dir}")

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

	# Загрузка данных о зрачках
	pupil_data = pd.read_csv(pupil_path)

	# Запуск всех алгоритмов
	all_results = []

	for alg_name, segmenter in segmenters.items():
		result = run_segmentation_algorithm(
			str(pupil_path),
			alg_name,
			segmenter
		)
		all_results.append(result)

		# Создание индивидуального графика
		if CREATE_PLOTS and result['error'] is None:
			plot_file = output_dir / f"segmentation_{alg_name.lower()}.html"
			create_segmentation_plot(
				pupil_data.copy(),
				result['fixations'],
				result['saccades'],
				alg_name,
				str(plot_file)
			)

	# Создание сравнительного графика
	if CREATE_PLOTS:
		comparison_file = output_dir / "segmentation_comparison.html"
		create_comparison_plot(pupil_data.copy(), all_results, str(comparison_file))

	# Сохранение результатов сегментации
	for result in all_results:
		if result['error'] is None:
			alg_name = result['algorithm']
			result['fixations'].to_csv(output_dir / f"fixations_{alg_name.lower()}.csv", index=False)
			result['saccades'].to_csv(output_dir / f"saccades_{alg_name.lower()}.csv", index=False)

	print(f"\n{'=' * 60}")
	print("Все алгоритмы завершены!")
	print(f"Результаты сохранены в: {output_dir}")
	print(f"{'=' * 60}")

	# Вывод статистики
	print(f"\nСтатистика сегментации:")
	for result in all_results:
		if result['error'] is None:
			print(f"  {result['algorithm']}: фиксаций={len(result['fixations'])}, саккад={len(result['saccades'])}")
		else:
			print(f"  {result['algorithm']}: ОШИБКА - {result['error']}")

	return 0


if __name__ == '__main__':
	sys.exit(main())