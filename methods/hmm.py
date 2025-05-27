import pandas as pd
import numpy as np
from hmmlearn import hmm
from scipy.signal import savgol_filter


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

# Сглаживание
df['x_smooth'] = savgol_filter(df['x'], window_length=5, polyorder=2)
df['y_smooth'] = savgol_filter(df['y'], window_length=5, polyorder=2)

# Вычисление скорости
dt = df['time'].diff().fillna(1 / 30)
dx = df['x_smooth'].diff().fillna(0)
dy = df['y_smooth'].diff().fillna(0)
velocity = np.sqrt(dx**2 + dy**2) / dt

# Подготовка данных для HMM
X = np.column_stack([velocity, df['x_smooth'], df['y_smooth']])
X = X[~np.isnan(X).any(axis=1)]  # Удаляем NaN

# Обучение HMM
model = hmm.GaussianHMM(n_components=2, covariance_type='diag', n_iter=100)
model.fit(X)
states = model.predict(X)

# Определение фиксаций и саккад (фиксации — низкая скорость)
mean_velocity = [np.mean(X[states == i, 0]) for i in range(2)]
fixation_state = np.argmin(mean_velocity)

# Группировка фиксаций
fixations = []
saccades = []
current_fixation = None
for i in range(len(df)):
    if i >= len(states):
        break
    if states[i] == fixation_state:
        if current_fixation is None:
            current_fixation = {'start_time': df['time'].iloc[i], 'x': [], 'y': []}
        current_fixation['x'].append(df['x_smooth'].iloc[i])
        current_fixation['y'].append(df['y_smooth'].iloc[i])
    else:
        if current_fixation:
            duration = df['time'].iloc[i] - current_fixation['start_time']
            if duration >= 0.1:
                fixations.append({
                    'start_time': current_fixation['start_time'],
                    'end_time': df['time'].iloc[i],
                    'duration': duration,
                    'x_center': np.mean(current_fixation['x']),
                    'y_center': np.mean(current_fixation['y'])
                })
            current_fixation = None
        if i > 0 and states[i-1] == fixation_state:
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

if current_fixation:
    duration = df['time'].iloc[-1] - current_fixation['start_time']
    if duration >= 0.1:
        fixations.append({
            'start_time': current_fixation['start_time'],
            'end_time': df['time'].iloc[-1],
            'duration': duration,
            'x_center': np.mean(current_fixation['x']),
            'y_center': np.mean(current_fixation['y'])
        })

# Сохранение
fixations_df = pd.DataFrame(fixations)
saccades_df = pd.DataFrame(saccades)
fixations_df.to_csv('fixations_hmm_8.csv', index=False)
saccades_df.to_csv('saccades_hmm_8.csv', index=False)
print("Результаты сохранены: fixations_hmm_8.csv, saccades_hmm_8.csv")