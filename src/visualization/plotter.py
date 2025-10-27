"""
Модуль визуализации движений глаз

Предоставляет возможности визуализации данных о движениях глаз,
включая построение графиков сырых данных, фиксаций и саккад.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional


class EyeMovementPlotter:
    """
    Построитель графиков для данных о движениях глаз.
    
    Создает интерактивные HTML графики с использованием библиотеки Plotly.
    """
    
    @staticmethod
    def parse_time(filename: str) -> float:
        """
        Парсинг времени из имени файла.
        
        Извлекает временную метку из имени файла в формате 'frame_0_00_00.jpg'.
        """
        try:
            parts = filename.replace('frame_', '').replace('.jpg', '').split('_')
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            microseconds = int(parts[3]) if len(parts) > 3 else 0
            return seconds + microseconds / 1_000_000
        except:
            return 0.0
    
    def plot_raw_data(self, pupil_csv: str, output_file: str = "eye_movement_plot.html"):
        """
        Построение графика сырых данных о движении зрачков.
        
        Создает интерактивный HTML график с двумя подграфиками:
        горизонтальное и вертикальное движение глаз.
        
        Args:
            pupil_csv: Путь к CSV файлу с данными о зрачках
            output_file: Путь к выходному HTML файлу
        """
        # Загрузка данных
        df = pd.read_csv(pupil_csv)
        
        # Извлечение времени из имени файла
        if 'filename' in df.columns:
            df['time'] = df['filename'].apply(self.parse_time)
        else:
            # Если нет имени файла - используем индексы
            df['time'] = range(len(df))
        
        # Вычисление средних координат между левым и правым зрачками
        avg_x = (df['left_x'] + df['right_x']) / 2
        avg_y = (df['left_y'] + df['right_y']) / 2
        
        # Создание фигуры с двумя подграфиками
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Eye Movement: Horizontal (x)", "Eye Movement: Vertical (y)"),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Добавление графика горизонтального движения (X)
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=avg_x,
                mode='lines',
                name='Horizontal (x)',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Добавление графика вертикального движения (Y)
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=avg_y,
                mode='lines',
                name='Vertical (y)',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
        
        # Настройка подписей осей
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Position (pixels)", row=1, col=1)
        fig.update_yaxes(title_text="Position (pixels)", row=2, col=1)
        
        fig.update_layout(
            title="Eye Movement Analysis",
            showlegend=True,
            height=600,
            width=800
        )
        
        fig.write_html(output_file)
        print(f"Plot saved to {output_file}")
    
    def plot_with_segmentation(self, pupil_csv: str, fixations_csv: str, 
                               saccades_csv: str, output_file: str = "eye_movement_segmented.html"):
        """
        Построение графика данных о движении глаз с отмеченными фиксациями и саккадами.
        
        Создает интерактивный HTML график, где точки окрашены разными цветами
        в зависимости от типа движения глаз: фиксация, саккада или другое.
        
        Args:
            pupil_csv: Путь к CSV файлу с данными о зрачках
            fixations_csv: Путь к CSV файлу с фиксациями
            saccades_csv: Путь к CSV файлу с саккадами
            output_file: Путь к выходному HTML файлу
        """
        # Загрузка всех данных
        pupil_df = pd.read_csv(pupil_csv)
        fixations_df = pd.read_csv(fixations_csv)
        saccades_df = pd.read_csv(saccades_csv)
        
        # Обработка данных о зрачках
        pupil_df['time'] = pupil_df['filename'].apply(self.parse_time)
        pupil_df['x'] = (pupil_df['left_x'] + pupil_df['right_x']) / 2
        pupil_df['y'] = (pupil_df['left_y'] + pupil_df['right_y']) / 2
        pupil_df['type'] = 'Other'  # По умолчанию все точки - "другие"
        
        # Отметка фиксаций
        for _, fixation in fixations_df.iterrows():
            mask = (pupil_df['time'] >= fixation['start_time']) & (pupil_df['time'] <= fixation['end_time'])
            pupil_df.loc[mask, 'type'] = 'Fixation'
        
        # Отметка саккад
        for _, saccade in saccades_df.iterrows():
            mask = (pupil_df['time'] >= saccade['start_time']) & (pupil_df['time'] <= saccade['end_time'])
            pupil_df.loc[mask, 'type'] = 'Saccade'
        
        # Разделение данных по типу движения
        fixation_df = pupil_df[pupil_df['type'] == 'Fixation']
        saccade_df = pupil_df[pupil_df['type'] == 'Saccade']
        other_df = pupil_df[pupil_df['type'] == 'Other']
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Horizontal movement (x)", "Vertical movement (y)")
        )
        
        # Fixations
        fig.add_trace(
            go.Scatter(
                x=fixation_df['time'],
                y=fixation_df['x'],
                mode='markers',
                name='Fixation',
                marker=dict(color='green', size=5)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=fixation_df['time'],
                y=fixation_df['y'],
                mode='markers',
                name='Fixation',
                marker=dict(color='green', size=5),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Saccades
        fig.add_trace(
            go.Scatter(
                x=saccade_df['time'],
                y=saccade_df['x'],
                mode='markers',
                name='Saccade',
                marker=dict(color='red', size=3)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=saccade_df['time'],
                y=saccade_df['y'],
                mode='markers',
                name='Saccade',
                marker=dict(color='red', size=3),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Other
        fig.add_trace(
            go.Scatter(
                x=other_df['time'],
                y=other_df['x'],
                mode='markers',
                name='Other',
                marker=dict(color='gray', size=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=other_df['time'],
                y=other_df['y'],
                mode='markers',
                name='Other',
                marker=dict(color='gray', size=2),
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="X (pixels)", row=1, col=1)
        fig.update_yaxes(title_text="Y (pixels)", row=2, col=1)
        
        fig.update_layout(
            title="Eye Movement with Fixations and Saccades",
            showlegend=True,
            height=800
        )
        
        fig.write_html(output_file)
        print(f"Plot saved to {output_file}")

