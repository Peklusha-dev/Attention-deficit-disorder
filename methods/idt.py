import pandas as pd
import numpy as np


# Парсинг времени
def parse_time(filename):
	parts = filename.replace('frame_', '').replace('.jpg', '').split('_')
	seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
	microseconds = int(parts[3]) if len(parts) > 3 else 0
	return seconds + microseconds / 1_000_000


# Загрузка данных
input_csv = 'pupil_data_8.csv'  # Замените на ваш CSV
df = pd.read_csv(input_csv)
df['time'] = df['filename'].apply(parse_time)
df['x'] = (df['left_x'] + df['right_x']) / 2
df['y'] = (df['left_y'] + df['right_y']) / 2

# Пороги
dispersion_threshold = 7  # Пикс
window_duration = 0.2  # Сек
min_fixation_duration = 0.1  # Сек
fps = 30  # Предполагаем
window_size = int(window_duration * fps)


# Функция для вычисления дисперсии
def dispersion(x, y):
	return np.max(x) - np.min(x) + np.max(y) - np.min(y)


# Детекция фиксаций
fixations = []
saccades = []
i = 0
while i < len(df) - window_size:
	window = df.iloc[i:i + window_size]
	if dispersion(window['x'], window['y']) < dispersion_threshold:
		start_idx = i
		start_time = window['time'].iloc[0]
		while i < len(df) - window_size and dispersion(df['x'].iloc[i:i + window_size],
		                                               df['y'].iloc[i:i + window_size]) < dispersion_threshold:
			i += 1
		end_idx = i + window_size - 1
		end_time = df['time'].iloc[end_idx]

		duration = end_time - start_time
		if duration >= min_fixation_duration:
			fixations.append({
				'start_time': start_time,
				'end_time': end_time,
				'duration': duration,
				'x_center': np.mean(df['x'].iloc[start_idx:end_idx + 1]),
				'y_center': np.mean(df['y'].iloc[start_idx:end_idx + 1])
			})
	else:
		i += 1

# Детекция саккад
for j in range(1, len(fixations)):
	saccades.append({
		'start_time': fixations[j - 1]['end_time'],
		'end_time': fixations[j]['start_time'],
		'start_x': fixations[j - 1]['x_center'],
		'start_y': fixations[j - 1]['y_center'],
		'end_x': fixations[j]['x_center'],
		'end_y': fixations[j]['y_center'],
		'amplitude': np.sqrt((fixations[j]['x_center'] - fixations[j - 1]['x_center']) ** 2 +
		                     (fixations[j]['y_center'] - fixations[j - 1]['y_center']) ** 2)
	})

# Сохранение
fixations_df = pd.DataFrame(fixations)
saccades_df = pd.DataFrame(saccades)
fixations_df.to_csv('fixations_idt_8.csv', index=False)
saccades_df.to_csv('saccades_idt_8.csv', index=False)
print("Результаты сохранены: fixations_idt_8.csv, saccades_idt_8.csv")