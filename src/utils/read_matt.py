import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import loadmat


def analyze_mat_file(file_path, category):
	"""Анализирует .mat файл и возвращает информацию о его структуре"""
	print(f"\n{'=' * 60}")
	print(f"АНАЛИЗ ФАЙЛА: {os.path.basename(file_path)}")
	print(f"КАТЕГОРИЯ: {category}")
	print(f"{'=' * 60}")

	try:
		# Загрузка файла
		mat_data = loadmat(file_path)

		# Основная информация
		print("Ключи в .mat файле:", [key for key in mat_data.keys() if not key.startswith('__')])

		# Анализ ETdata
		if 'ETdata' in mat_data:
			et_data = mat_data['ETdata'][0, 0]
			raw_data = et_data[0]  # Основной массив данных

			print(f"Размер основного массива: {raw_data.shape}")
			print(f"Количество записей: {len(raw_data)}")

			# Информация о частоте дискретизации
			if len(et_data) > 4:
				sampling_rate = et_data[4]
				print(f"Частота дискретизации: {sampling_rate[0, 0]} Гц")
				print(f"Длительность записи: {len(raw_data) / sampling_rate[0, 0]:.2f} секунд")

			# Информация о разрешении экрана
			if len(et_data) > 2:
				screen_res = et_data[2]
				print(f"Разрешение экрана: {screen_res[0, 0]}×{screen_res[0, 1]}")

			# Пример первых 3 записей
			print("\nПервые 3 записи данных:")
			for i in range(min(3, len(raw_data))):
				print(f"  Запись {i}: {raw_data[i]}")

		# Анализ results
		if 'results' in mat_data:
			results = mat_data['results'][0, 0]
			print(f"\nРезультаты разметки:")
			print(f"  Коды событий: {results[0]}")
			if len(results) > 1:
				print(f"  Комментарий: {results[1][0][:100]}...")  # Первые 100 символов

		# Анализ fpData
		if 'fpData' in mat_data:
			fp_data = mat_data['fpData'][0, 0]
			print(f"\nМетаданные:")
			if len(fp_data) > 1:
				print(f"  Дата записи: {fp_data[1][0]}")
			print(f"  Инфо об испытуемом: {fp_data[0]}")

		return True

	except Exception as e:
		print(f"ОШИБКА при анализе файла: {e}")
		return False


def create_sample_visualization(file_path, output_name):
	"""Создает визуализацию для файла"""
	try:
		mat_data = loadmat(file_path)
		et_data = mat_data['ETdata'][0, 0]
		raw_data = et_data[0]

		x_coords = raw_data[:, 3]
		y_coords = raw_data[:, 4]

		plt.figure(figsize=(10, 8))

		# Траектория взгляда
		plt.subplot(2, 1, 1)
		plt.scatter(x_coords, y_coords, s=1, alpha=0.6, c=range(len(x_coords)), cmap='viridis')
		plt.colorbar(label='Время')
		plt.xlabel('X координата')
		plt.ylabel('Y координата')
		plt.title(f'Траектория взгляда - {output_name}')
		plt.gca().invert_yaxis()

		# Распределение по времени
		plt.subplot(2, 1, 2)
		time_seconds = np.arange(len(x_coords)) / 500  # Предполагаем 500 Гц
		plt.plot(time_seconds, x_coords, 'r-', alpha=0.7, label='X')
		plt.plot(time_seconds, y_coords, 'b-', alpha=0.7, label='Y')
		plt.xlabel('Время (секунды)')
		plt.ylabel('Координата')
		plt.legend()
		plt.title('Координаты во времени')

		plt.tight_layout()
		plt.savefig(f'{output_name}_plot.png', dpi=150, bbox_inches='tight')
		plt.close()

		print(f"Визуализация сохранена как: {output_name}_plot.png")

	except Exception as e:
		print(f"Ошибка при создании визуализации: {e}")


# Основная программа
if __name__ == "__main__":
	base_path = r'C:\C++_projects\EyeMovementDetectorEvaluation\annotated_data\data used in the article'

	# Выбираем по одному файлу из каждой категории для анализа
	sample_files = [
		(os.path.join(base_path, 'dots', 'TH20_trial1_labelled_MN.mat'), 'dots'),
		(os.path.join(base_path, 'img', 'TH34_img_Europe_labelled_MN.mat'), 'images'),
		(os.path.join(base_path, 'video', 'TH34_video_BergoDalbana_labelled_MN.mat'), 'video')
	]

	print("🚀 НАЧИНАЕМ АНАЛИЗ СТРУКТУРЫ ДАННЫХ")
	print("Будет проанализировано по одному файлу из каждой категории")

	for file_path, category in sample_files:
		if os.path.exists(file_path):
			success = analyze_mat_file(file_path, category)
			if success:
				# Создаем визуализацию для успешно проанализированных файлов
				output_name = f"{category}_{os.path.basename(file_path).replace('.mat', '')}"
				create_sample_visualization(file_path, output_name)
		else:
			print(f"\n❌ Файл не найден: {file_path}")

	print(f"\n{'=' * 60}")
	print("АНАЛИЗ ЗАВЕРШЕН!")
	print("Теперь у нас будет полное представление о структуре данных")