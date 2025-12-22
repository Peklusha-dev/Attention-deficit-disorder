import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


# Парсинг времени из имени файла
def parse_time(filename):
    parts = filename.replace('frame_', '').replace('.jpg', '').split('_')
    seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    microseconds = int(parts[3]) if len(parts) > 3 else 0
    return seconds + microseconds / 1_000_000

# Загрузка данных
input_csv = 'pupil_data_8.csv'
df = pd.read_csv(input_csv)

# Извлечение времени
df['time'] = df['filename'].apply(parse_time)

# Усреднение координат зрачков
df['x'] = (df['left_x'] + df['right_x']) / 2
df['y'] = (df['left_y'] + df['right_y']) / 2

# Сглаживание (опционально)
df['x_smooth'] = savgol_filter(df['x'], window_length=5, polyorder=2)
df['y_smooth'] = savgol_filter(df['y'], window_length=5, polyorder=2)

# Вычисление скорости
dt = df['time'].diff().fillna(1 / 30)  # Предполагаем 30 FPS
dx = df['x_smooth'].diff().fillna(0)
dy = df['y_smooth'].diff().fillna(0)
velocity = np.sqrt(dx**2 + dy**2) / dt  # Пикс/с

# Пороги
velocity_threshold = 110  # Пикс/с
min_fixation_duration = 0.1  # Сек

# Классификация
df['is_fixation'] = velocity < velocity_threshold

# Группировка фиксаций
fixations = []
saccades = []
current_fixation = None
for i in range(len(df)):
    if df['is_fixation'].iloc[i]:
        if current_fixation is None:
            current_fixation = {'start_time': df['time'].iloc[i], 'x': [], 'y': []}
        current_fixation['x'].append(df['x_smooth'].iloc[i])
        current_fixation['y'].append(df['y_smooth'].iloc[i])
    else:
        if current_fixation:
            duration = df['time'].iloc[i] - current_fixation['start_time']
            if duration >= min_fixation_duration:
                fixations.append({
                    'start_time': current_fixation['start_time'],
                    'end_time': df['time'].iloc[i],
                    'duration': duration,
                    'x_center': np.mean(current_fixation['x']),
                    'y_center': np.mean(current_fixation['y'])
                })
            current_fixation = None
        if i > 0 and df['is_fixation'].iloc[i-1]:
            saccades.append({
                'start_time': df['time'].iloc[i-1],
                'end_time': df['time'].iloc[i],
                'start_x': df['x_smooth'].iloc[i-1],
                'start_y': df['y_smooth'].iloc[i-1],
                'end_x': df['x_smooth'].iloc[i],
                'end_y': df['y_smooth'].iloc[i],
                'amplitude': np.sqrt((df['x_smooth'].iloc[i] - df['x_smooth'].iloc[i-1])**2 +
                                     (df['y_smooth'].iloc[i] - df['y_smooth'].iloc[i-1])**2)
            })

# Сохранение последней фиксации
if current_fixation:
    duration = df['time'].iloc[-1] - current_fixation['start_time']
    if duration >= min_fixation_duration:
        fixations.append({
            'start_time': current_fixation['start_time'],
            'end_time': df['time'].iloc[-1],
            'duration': duration,
            'x_center': np.mean(current_fixation['x']),
            'y_center': np.mean(current_fixation['y'])
        })

# Сохранение результатов
fixations_df = pd.DataFrame(fixations)
saccades_df = pd.DataFrame(saccades)
fixations_df.to_csv('fixations_ivt_8.csv', index=False)
saccades_df.to_csv('saccades_ivt_8.csv', index=False)
print("Результаты сохранены: fixations_ivt_8.csv, saccades_ivt_8.csv")