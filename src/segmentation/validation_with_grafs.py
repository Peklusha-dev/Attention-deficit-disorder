import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os


def load_single_file(file_path):
	"""Загрузка одного файла с данными"""
	data = scipy.io.loadmat(file_path)
	et_data = data['ETdata'][0, 0]
	raw_data = et_data[0]  # Основной массив

	return {
		'x': raw_data[:, 3],  # X-координаты
		'y': raw_data[:, 4],  # Y-координаты
		'labels': raw_data[:, 5],  # Разметка экспертов (1-6)
		'timestamps': raw_data[:, 0],
		'sampling_rate': 500,
		'filename': os.path.basename(file_path)
	}


def corrected_ivt_algorithm(x_coords, y_coords, velocity_threshold=200):
	"""Исправленный I-VT алгоритм"""
	dx = np.diff(x_coords, prepend=x_coords[0])
	dy = np.diff(y_coords, prepend=y_coords[0])
	dt = 1.0 / 500  # 500 Гц
	velocity = np.sqrt(dx ** 2 + dy ** 2) / dt  # пиксели/секунду

	predictions = (velocity < velocity_threshold).astype(int)
	return predictions, velocity


def compute_angle_trajectory(x, y, center_x=512, center_y=384):
	"""Вычисление угла/отклонения от центра"""
	# Отклонение от центра экрана
	dx = x - center_x
	dy = y - center_y

	# Угол в радианах (atan2 дает угол от -π до π)
	angle = np.arctan2(dy, dx)

	# Расстояние от центра (амплитуда)
	distance = np.sqrt(dx ** 2 + dy ** 2)

	return angle, distance


def plot_simple_comparison(dataset, predictions, algorithm_name):
	"""Упрощенная визуализация только с 4 графиками"""

	# Вычисляем угол/траекторию
	angle, distance = compute_angle_trajectory(dataset['x'], dataset['y'])
	time_sec = np.arange(len(dataset['x'])) / dataset['sampling_rate']

	# Создаем фигуру с 2x2 графиками
	fig, axes = plt.subplots(2, 2, figsize=(16, 10))
	fig.suptitle(f'Сравнение разметки: {dataset["filename"]}\nАлгоритм: {algorithm_name}',
	             fontsize=16, fontweight='bold')

	# Цвета для разметки экспертов
	colors = ['red', 'green', 'yellow', 'pink', 'black', 'cyan']
	labels_names = ['Фиксация', 'Саккада', 'PSO', 'Плавное слежение', 'Моргание', 'Неопределенное']

	# 1. Траектория взгляда с разметкой экспертов (координаты)
	ax1 = axes[0, 0]
	for label in range(1, 7):
		mask = dataset['labels'] == label
		if np.any(mask):
			ax1.scatter(dataset['x'][mask], dataset['y'][mask],
			            c=colors[label - 1], s=8, label=labels_names[label - 1], alpha=0.7)

	ax1.set_title('Траектория взгляда - Разметка экспертов')
	ax1.set_xlabel('X координата (пиксели)')
	ax1.set_ylabel('Y координата (пиксели)')
	ax1.legend(fontsize=8)
	ax1.invert_yaxis()
	ax1.grid(True, alpha=0.3)

	# 2. Траектория взгляда с разметкой алгоритма (координаты)
	ax2 = axes[0, 1]
	# Фиксации алгоритма (красные)
	fixation_mask = predictions == 1
	ax2.scatter(dataset['x'][fixation_mask], dataset['y'][fixation_mask],
	            c='red', s=8, label='Фиксации (алгоритм)', alpha=0.7)
	# Не фиксации (синие)
	non_fixation_mask = predictions == 0
	ax2.scatter(dataset['x'][non_fixation_mask], dataset['y'][non_fixation_mask],
	            c='blue', s=8, label='Не фиксации (алгоритм)', alpha=0.7)

	ax2.set_title('Траектория взгляда - Разметка алгоритма')
	ax2.set_xlabel('X координата (пиксели)')
	ax2.set_ylabel('Y координата (пиксели)')
	ax2.legend(fontsize=8)
	ax2.invert_yaxis()
	ax2.grid(True, alpha=0.3)

	# 3. Угловая траектория с разметкой экспертов
	ax3 = axes[1, 0]
	for label in range(1, 7):
		mask = dataset['labels'] == label
		if np.any(mask):
			ax3.scatter(time_sec[mask], angle[mask],
			            c=colors[label - 1], s=8, label=labels_names[label - 1], alpha=0.7)

	ax3.set_title('Угловая траектория - Разметка экспертов')
	ax3.set_xlabel('Время (секунды)')
	ax3.set_ylabel('Угол (радианы)')
	ax3.legend(fontsize=8)
	ax3.grid(True, alpha=0.3)

	# 4. Угловая траектория с разметкой алгоритма
	ax4 = axes[1, 1]
	# Фиксации алгоритма (красные)
	ax4.scatter(time_sec[fixation_mask], angle[fixation_mask],
	            c='red', s=8, label='Фиксации (алгоритм)', alpha=0.7)
	# Не фиксации (синие)
	ax4.scatter(time_sec[non_fixation_mask], angle[non_fixation_mask],
	            c='blue', s=8, label='Не фиксации (алгоритм)', alpha=0.7)

	ax4.set_title('Угловая траектория - Разметка алгоритма')
	ax4.set_xlabel('Время (секунды)')
	ax4.set_ylabel('Угол (радианы)')
	ax4.legend(fontsize=8)
	ax4.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.show()

	# Вывод статистики
	print(f"\n{'=' * 60}")
	print(f"СТАТИСТИКА ДЛЯ: {dataset['filename']}")
	print(f"{'=' * 60}")
	print(f"Всего samples: {len(dataset['x'])}")
	print(f"Длительность: {len(dataset['x']) / 500:.2f} секунд")

	# Распределение меток экспертов
	unique, counts = np.unique(dataset['labels'], return_counts=True)
	print(f"\nРАСПРЕДЕЛЕНИЕ МЕТОК ЭКСПЕРТОВ:")
	for label, count in zip(unique, counts):
		label_names = {1: 'Фиксации', 2: 'Саккады', 3: 'PSO', 4: 'Плавное слежение',
		               5: 'Моргания', 6: 'Неопределенные'}
		percentage = (count / len(dataset['labels'])) * 100
		print(f"  {label_names.get(label, f'Метка {label}')}: {count} samples ({percentage:.1f}%)")

	# Статистика алгоритма
	print(f"\nСТАТИСТИКА АЛГОРИТМА:")
	print(f"  Предсказано фиксаций: {np.sum(predictions)} ({np.mean(predictions) * 100:.1f}%)")


# Анализ конкретных файлов
if __name__ == "__main__":
	base_path = r'C:\C++_projects\EyeMovementDetectorEvaluation\annotated_data\data used in the article'

	# Выбираем по одному файлу из каждой категории
	files_to_analyze = [
		(os.path.join(base_path, 'dots', 'TH20_trial1_labelled_MN.mat'), 'DOTS'),
		(os.path.join(base_path, 'img', 'TH34_img_Europe_labelled_MN.mat'), 'IMAGES'),
		(os.path.join(base_path, 'video', 'TH34_video_BergoDalbana_labelled_MN.mat'), 'VIDEO')
	]

	for file_path, category in files_to_analyze:
		if os.path.exists(file_path):
			print(f"\n{'#' * 80}")
			print(f"АНАЛИЗИРУЕМ: {category}")
			print(f"Файл: {os.path.basename(file_path)}")
			print(f"{'#' * 80}")

			# Загружаем данные
			dataset = load_single_file(file_path)

			# Запускаем алгоритм
			predictions, velocity = corrected_ivt_algorithm(dataset['x'], dataset['y'],
			                                                velocity_threshold=200)

			# Визуализируем
			plot_simple_comparison(dataset, predictions, f"I-VT (200 px/s) - {category}")

		else:
			print(f"Файл не найден: {file_path}")