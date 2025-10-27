import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Парсинг времени из имени файла
def parse_time(filename):
    parts = filename.replace('frame_', '').replace('.jpg', '').split('_')
    seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    microseconds = int(parts[3]) if len(parts) > 3 else 0
    return seconds + microseconds / 1_000_000

# Загрузка данных
pupil_csv = 'pupil_data_8.csv'  # Замените на ваш CSV с координатами
fixations_csv = 'fixations_hmm_8.csv'  # Замените на ваш CSV с фиксациями
saccades_csv = 'saccades_hmm_8.csv'  # Замените на ваш CSV с саккадами

# Чтение сырых координат
try:
    pupil_df = pd.read_csv(pupil_csv)
    if not all(col in pupil_df.columns for col in ['filename', 'left_x', 'left_y', 'right_x', 'right_y']):
        raise ValueError("CSV должен содержать колонки: filename, left_x, left_y, right_x, right_y")
    pupil_df['time'] = pupil_df['filename'].apply(parse_time)
except Exception as e:
    print(f"Ошибка при загрузке {pupil_csv}: {e}")
    exit()

# Вычисление средней точки между зрачками
pupil_df['avg_x'] = (pupil_df['left_x'] + pupil_df['right_x']) / 2
pupil_df['avg_y'] = (pupil_df['left_y'] + pupil_df['right_y']) / 2

# Чтение фиксаций и саккад
try:
    fixations_df = pd.read_csv(fixations_csv)
    if not all(col in fixations_df.columns for col in ['start_time', 'end_time']):
        raise ValueError("fixations_csv должен содержать: start_time, end_time")
except FileNotFoundError:
    print(f"Файл {fixations_csv} не найден. Создайте его с помощью алгоритма (например, I-VT).")
    exit()
except Exception as e:
    print(f"Ошибка при загрузке {fixations_csv}: {e}")
    exit()

try:
    saccades_df = pd.read_csv(saccades_csv)
    if not all(col in saccades_df.columns for col in ['start_time', 'end_time']):
        raise ValueError("saccades_csv должен содержать: start_time, end_time")
except FileNotFoundError:
    print(f"Файл {saccades_csv} не найден. Создайте его с помощью алгоритма (например, I-VT).")
    exit()
except Exception as e:
    print(f"Ошибка при загрузке {saccades_csv}: {e}")
    exit()

# Классификация точек
pupil_df['type'] = 'Другое'  # По умолчанию "другое"

# Отметка фиксаций
for _, fixation in fixations_df.iterrows():
    mask = (pupil_df['time'] >= fixation['start_time']) & (pupil_df['time'] <= fixation['end_time'])
    pupil_df.loc[mask, 'type'] = 'Фиксация'

# Отметка саккад
for _, saccade in saccades_df.iterrows():
    mask = (pupil_df['time'] >= saccade['start_time']) & (pupil_df['time'] <= saccade['end_time'])
    pupil_df.loc[mask, 'type'] = 'Саккада'

# Разделение данных по типу
fixation_df = pupil_df[pupil_df['type'] == 'Фиксация']
saccade_df = pupil_df[pupil_df['type'] == 'Саккада']
other_df = pupil_df[pupil_df['type'] == 'Другое']

# Создание подграфиков
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=("Горизонтальное движение (x)", "Вертикальное движение (y)")
)

# Функция для добавления точек
def add_points(df, label, color, row, x_col, y_col, showlegend=True):
    if not df.empty:
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df[x_col],
                mode='markers',
                name=label,
                marker=dict(color=color, size=6),
                hovertemplate=f'Type: {label}<br>Time: %{{x:.3f}} s<br>{y_col.replace("_", " ").title()}: %{{y:.1f}} px',
                showlegend=showlegend
            ),
            row=row, col=1
        )

# Точки для средней координаты
# Фиксации
add_points(
    fixation_df,
    label='Средняя (Фиксация)',
    color='green',
    row=1,
    x_col='avg_x',
    y_col='Avg X'
)
add_points(
    fixation_df,
    label='Средняя (Фиксация)',
    color='green',
    row=2,
    x_col='avg_y',
    y_col='Avg Y',
    showlegend=False
)

# Саккады
add_points(
    saccade_df,
    label='Средняя (Саккада)',
    color='red',
    row=1,
    x_col='avg_x',
    y_col='Avg X'
)
add_points(
    saccade_df,
    label='Средняя (Саккада)',
    color='red',
    row=2,
    x_col='avg_y',
    y_col='Avg Y',
    showlegend=False
)

# Другое
add_points(
    other_df,
    label='Средняя (Другое)',
    color='gray',
    row=1,
    x_col='avg_x',
    y_col='Avg X'
)
add_points(
    other_df,
    label='Средняя (Другое)',
    color='gray',
    row=2,
    x_col='avg_y',
    y_col='Avg Y',
    showlegend=False
)

# Настройка осей
fig.update_xaxes(title_text="Время (с)", row=2, col=1)
fig.update_yaxes(title_text="X (пиксели)", row=1, col=1)
fig.update_yaxes(title_text="Y (пиксели)", row=2, col=1)

# Настройка макета
fig.update_layout(
    title="Движение глаз: средние точки с фиксациями и саккадами",
    showlegend=True,
    hovermode='closest',
    height=800,
    template='plotly_white'
)

# Сохранение
fig.write_html('eye_movement_scatter_avg_8_hmm.html')
print("График сохранён в eye_movement_scatter_avg_8_hmm.html")