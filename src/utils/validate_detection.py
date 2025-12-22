"""
Интерактивная разметка траектории движений зрачков

Скрипт для построения графика траектории движений зрачков с возможностью
интерактивной разметки диапазонов саккад и фиксаций.

Использование:
1. Укажите путь к видео (VIDEO_PATH) или папке с изображениями (IMAGE_FOLDER)
2. Выберите тип детектора (DETECTOR_TYPE): "basic" или "head_compensated"
3. Запустите скрипт: python examples/interactive_trajectory_marker.py
4. Откройте браузер по адресу http://127.0.0.1:8050
5. Выделите диапазон на графике (используйте инструмент выделения в панели инструментов)
6. Выберите тип события (Саккада или Фиксация)
7. Нажмите "Сохранить выделенный диапазон"
8. Нажмите "Сохранить все в CSV" для сохранения разметки

Результаты сохраняются в CSV файл с колонками: type, start_time, end_time
"""

import sys
import warnings
import os
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context

# Подавление предупреждений
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Добавление src в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import PupilDetector, PupilHeadDetector

# Настройки
#VIDEO_PATH = None
IMAGE_FOLDER = None
VIDEO_PATH = "../dataset/eyes_movies/маятник.MOV"  # Путь к видео или None
#IMAGE_FOLDER = "../dataset/eyes_movies/маятник"  # Путь к папке с изображениями или None (если указан VIDEO_PATH)
OUTPUT_PUPIL_CSV = "../маятник.csv"
OUTPUT_ANNOTATIONS_CSV = "../маятник.csv"
DETECTOR_TYPE = "basic"  # "basic" или "head_compensated"


def parse_time(filename: str) -> float:
    """Парсинг времени из имени файла."""
    try:
        parts = filename.replace('frame_', '').replace('.jpg', '').split('_')
        if len(parts) >= 3:
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            microseconds = int(parts[3]) if len(parts) > 3 else 0
            return seconds + microseconds / 1_000_000
        return 0.0
    except:
        return 0.0


def process_data_for_trajectory():
    """Обработка видео или папки с изображениями и сохранение данных о зрачках."""
    if DETECTOR_TYPE == "basic":
        detector = PupilDetector()
    else:
        detector = PupilHeadDetector()

    if VIDEO_PATH and Path(VIDEO_PATH).exists():
        print("Обработка видео...")
        detector.process_video(VIDEO_PATH, OUTPUT_PUPIL_CSV, show_result=False)
    elif IMAGE_FOLDER and Path(IMAGE_FOLDER).exists():
        print("Обработка папки с изображениями...")
        detector.process_image_folder(IMAGE_FOLDER, OUTPUT_PUPIL_CSV)
    else:
        raise ValueError("Укажите VIDEO_PATH или IMAGE_FOLDER")

    print(f"Данные сохранены в {OUTPUT_PUPIL_CSV}")
    return OUTPUT_PUPIL_CSV


def load_pupil_data(csv_path):
    """Загрузка данных о зрачках."""
    df = pd.read_csv(csv_path)

    if 'filename' in df.columns:
        df['time'] = df['filename'].apply(parse_time)
    else:
        df['time'] = range(len(df))

    df['x'] = (df['left_x'] + df['right_x']) / 2
    df['y'] = (df['left_y'] + df['right_y']) / 2

    return df


def create_trajectory_plot(df, annotations_df=None):
    """Создание графика траектории."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Горизонтальное движение (X)", "Вертикальное движение (Y)"),
        shared_xaxes=True,
        vertical_spacing=0.1
    )

    # График горизонтального движения
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['x'],
            mode='lines+markers',
            name='X координата',
            line=dict(color='blue', width=1),
            marker=dict(size=3, color='blue'),
            hovertemplate='Время: %{x:.3f}с<br>X: %{y:.1f}px<extra></extra>'
        ),
        row=1, col=1
    )

    # График вертикального движения
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['y'],
            mode='lines+markers',
            name='Y координата',
            line=dict(color='orange', width=1),
            marker=dict(size=3, color='orange'),
            hovertemplate='Время: %{x:.3f}с<br>Y: %{y:.1f}px<extra></extra>'
        ),
        row=2, col=1
    )

    # Добавление разметки саккад и фиксаций
    if annotations_df is not None and len(annotations_df) > 0:
        for _, event in annotations_df.iterrows():
            # Безопасная проверка наличия столбца 'type'
            event_type = event.get('type', 'unknown')
            color = 'red' if event_type == 'saccade' else 'green'
            alpha = 0.2

            # Добавление вертикальных областей для обоих графиков
            for row in [1, 2]:
                fig.add_vrect(
                    x0=event['start_time'],
                    x1=event['end_time'],
                    fillcolor=color,
                    opacity=alpha,
                    line_width=0,
                    row=row, col=1,
                    annotation_text=event_type,
                    annotation_position="top left"
                )

    fig.update_xaxes(title_text="Время (секунды)", row=2, col=1)
    fig.update_yaxes(title_text="Позиция (пиксели)", row=1, col=1)
    fig.update_yaxes(title_text="Позиция (пиксели)", row=2, col=1)

    fig.update_layout(
        title="Траектория движений зрачков",
        height=800,
        dragmode='select',
        hovermode='closest'
    )

    return fig


# Инициализация Dash приложения
app = dash.Dash(__name__)

# Загрузка данных
if not Path(OUTPUT_PUPIL_CSV).exists():
    process_data_for_trajectory()

df = load_pupil_data(OUTPUT_PUPIL_CSV)

# Загрузка существующих аннотаций
annotations_df = pd.DataFrame(columns=['type', 'start_time', 'end_time'])
if Path(OUTPUT_ANNOTATIONS_CSV).exists():
    annotations_df = pd.read_csv(OUTPUT_ANNOTATIONS_CSV)

# Создание начального графика
fig = create_trajectory_plot(df, annotations_df)

# Макет приложения
app.layout = html.Div([
    html.H2("Интерактивная разметка траектории движений зрачков",
            style={'textAlign': 'center', 'marginBottom': '20px'}),

    html.Div([
        html.Label("Тип события:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.RadioItems(
            id='event-type',
            options=[
                {'label': 'Саккада', 'value': 'saccade'},
                {'label': 'Фиксация', 'value': 'fixation'}
            ],
            value='saccade',
            inline=True,
            style={'marginBottom': '20px'}
        ),
    ], style={'marginBottom': '20px', 'padding': '10px', 'border': '1px solid #ccc'}),

    html.Div([
        html.Button('Сохранить выделенный диапазон', id='save-button', n_clicks=0,
                   style={'marginRight': '10px', 'padding': '10px 20px', 'fontSize': '14px'}),
        html.Button('Удалить последнюю аннотацию', id='delete-button', n_clicks=0,
                   style={'marginRight': '10px', 'padding': '10px 20px', 'fontSize': '14px'}),
        html.Button('Сохранить все в CSV', id='export-button', n_clicks=0,
                   style={'padding': '10px 20px', 'fontSize': '14px'})
    ], style={'marginBottom': '20px'}),

    html.Div(id='status-message', style={'marginBottom': '20px', 'padding': '10px',
                                         'backgroundColor': '#f0f0f0', 'borderRadius': '5px'}),

    dcc.Graph(
        id='trajectory-plot',
        figure=fig,
        config={
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['select2d', 'lasso2d']
        }
    ),

    html.Div([
        html.H5("Текущие аннотации:", style={'marginTop': '20px'}),
        html.Div(id='annotations-list', style={'padding': '10px', 'backgroundColor': '#f9f9f9',
                                               'borderRadius': '5px', 'maxHeight': '200px',
                                               'overflowY': 'auto'})
    ])
], style={'padding': '20px', 'maxWidth': '1400px', 'margin': '0 auto'})


@app.callback(
    [Output('trajectory-plot', 'figure'),
     Output('status-message', 'children'),
     Output('annotations-list', 'children')],
    [Input('save-button', 'n_clicks'),
     Input('delete-button', 'n_clicks'),
     Input('export-button', 'n_clicks')],
    [State('event-type', 'value'),
     State('trajectory-plot', 'selectedData')]
)
def update_plot(save_clicks, delete_clicks, export_clicks, event_type, selected_data):
    """Обработка действий пользователя."""
    global annotations_df

    ctx = callback_context
    if not ctx.triggered:
        return create_trajectory_plot(df, annotations_df), "", create_annotations_list()

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'save-button' and selected_data:
        # Получение диапазона времени из выделения
        start_time = None
        end_time = None

        # Проверка различных форматов выделения
        if 'range' in selected_data:
            if 'x' in selected_data['range']:
                x_range = selected_data['range']['x']
                start_time = x_range[0]
                end_time = x_range[1]
        elif 'points' in selected_data and len(selected_data['points']) > 0:
            times = [p['x'] for p in selected_data['points'] if 'x' in p]
            if times:
                start_time = min(times)
                end_time = max(times)

        if start_time is None or end_time is None:
            return create_trajectory_plot(df, annotations_df), \
                   html.Div("Ошибка: выделите диапазон на графике (используйте инструмент выделения)",
                           style={'color': 'red'}), \
                   create_annotations_list()

        # Убеждаемся, что start_time < end_time
        if start_time > end_time:
            start_time, end_time = end_time, start_time

        # Добавление новой аннотации
        new_annotation = pd.DataFrame({
            'type': [event_type],
            'start_time': [start_time],
            'end_time': [end_time]
        })
        annotations_df = pd.concat([annotations_df, new_annotation], ignore_index=True)
        annotations_df = annotations_df.sort_values('start_time').reset_index(drop=True)

        message = html.Div(f"Добавлена {event_type}: {start_time:.3f}с - {end_time:.3f}с",
                          style={'color': 'green'})

    elif button_id == 'delete-button' and len(annotations_df) > 0:
        # Удаление последней аннотации
        annotations_df = annotations_df.iloc[:-1].reset_index(drop=True)
        message = html.Div("Последняя аннотация удалена", style={'color': 'orange'})

    elif button_id == 'export-button':
        # Сохранение в CSV
        annotations_df.to_csv(OUTPUT_ANNOTATIONS_CSV, index=False)
        message = html.Div(f"Аннотации сохранены в {OUTPUT_ANNOTATIONS_CSV}",
                          style={'color': 'green'})

    else:
        message = html.Div("Выделите диапазон на графике (используйте инструмент выделения) и нажмите 'Сохранить'",
                          style={'color': 'blue'})

    return create_trajectory_plot(df, annotations_df), message, create_annotations_list()


def create_annotations_list():
    """Создание списка текущих аннотаций."""
    if len(annotations_df) == 0:
        return html.Div("Нет аннотаций")

    items = []
    for idx, row in annotations_df.iterrows():
        # Безопасная проверка наличия столбца 'type'
        event_type = row.get('type', 'unknown')
        color = 'red' if event_type == 'saccade' else 'green'
        items.append(html.Div([
            html.Span(f"{event_type}: ", style={'font-weight': 'bold', 'color': color}),
            html.Span(f"{row['start_time']:.3f}с - {row['end_time']:.3f}с")
        ], className="mb-1"))

    return html.Div(items)


if __name__ == '__main__':
    print(f"Запуск интерактивного интерфейса...")
    print(f"Откройте браузер по адресу: http://127.0.0.1:8053")
    app.run(debug=True, port=8053)