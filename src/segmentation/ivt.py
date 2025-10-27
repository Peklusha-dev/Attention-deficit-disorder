"""
Алгоритм I-VT (I-Velocity-Threshold) для сегментации движений глаз

Классифицирует движения глаз как фиксации или саккады на основе порога скорости.
Если скорость ниже порога - это фиксация, иначе - саккада.
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, List


class IVTSegmenter:
    """
    Сегментатор I-VT для классификации движений глаз.
    
    Классифицирует движения глаз на фиксации и саккады на основе скорости.
    Фиксации определяются как периоды, когда скорость ниже заданного порога.
    """
    
    def __init__(self, velocity_threshold: float = 110, min_fixation_duration: float = 0.1, fps: float = 30):
        """
        Инициализация сегментатора I-VT.
        
        Args:
            velocity_threshold: Порог скорости в пикселях/секунду.
                               Если скорость выше - движение считается саккадой,
                               если ниже - фиксацией.
            min_fixation_duration: Минимальная длительность фиксации в секундах.
                                  Фиксации короче этого времени не сохраняются.
            fps: Частота кадров для расчета времени.
        """
        self.velocity_threshold = velocity_threshold
        self.min_fixation_duration = min_fixation_duration
        self.fps = fps
    
    @staticmethod
    def parse_time(filename: str) -> float:
        """
        Парсинг времени из имени файла в формате 'frame_0_00_00.jpg'.
        
        Args:
            filename: Имя файла с временной меткой
            
        Returns:
            Время в секундах
        """
        try:
            # Разбиение имени файла на части
            parts = filename.replace('frame_', '').replace('.jpg', '').split('_')
            # Преобразование в секунды
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            # Добавление микродекунд
            microseconds = int(parts[3]) if len(parts) > 3 else 0
            return seconds + microseconds / 1_000_000
        except:
            return 0.0
    
    def compute_velocity(self, df: pd.DataFrame) -> pd.Series:
        """
        Вычисление скорости движения глаз из данных о позиции.
        
        Вычисляет скорость движения как производную от позиции по времени.
        Использует формулу: v = sqrt((dx/dt)^2 + (dy/dt)^2)
        
        Args:
            df: DataFrame с колонками 'x_smooth', 'y_smooth', 'time'
            
        Returns:
            Series со значениями скорости в пикселях/секунду
        """
        # Интервал времени между кадрами
        dt = df['time'].diff().fillna(1 / self.fps)
        # Изменение по X
        dx = df['x_smooth'].diff().fillna(0)
        # Изменение по Y
        dy = df['y_smooth'].diff().fillna(0)
        # Вычисление скорости (пиксели/секунду)
        velocity = np.sqrt(dx**2 + dy**2) / dt
        return velocity
    
    def segment(self, input_csv: str, output_prefix: str = "eye_movement") -> Dict[str, pd.DataFrame]:
        """
        Сегментация движений глаз на фиксации и саккады.
        
        Метод загружает данные о зрачках, применяет сглаживание, вычисляет скорость
        и классифицирует каждый кадр как фиксацию или саккаду на основе порога скорости.
        
        Args:
            input_csv: Путь к CSV файлу с данными о зрачках
            output_prefix: Префикс для выходных файлов (не используется)
            
        Returns:
            Словарь с двумя DataFrame: 'fixations' и 'saccades'
        """
        # Загрузка данных из CSV файла
        df = pd.read_csv(input_csv)
        # Извлечение времени из имени файла
        df['time'] = df['filename'].apply(self.parse_time)
        # Средняя координата X между левым и правым зрачками
        df['x'] = (df['left_x'] + df['right_x']) / 2
        # Средняя координата Y между левым и правым зрачками
        df['y'] = (df['left_y'] + df['right_y']) / 2
        
        # Сглаживание данных с помощью фильтра Савгольца
        # window_length=5: окно из 5 точек
        # polyorder=2: полином 2-го порядка
        df['x_smooth'] = savgol_filter(df['x'], window_length=5, polyorder=2)
        df['y_smooth'] = savgol_filter(df['y'], window_length=5, polyorder=2)
        
        # Вычисление скорости движения
        velocity = self.compute_velocity(df)
        # Классификация: если скорость ниже порога - это фиксация
        df['is_fixation'] = velocity < self.velocity_threshold
        
        # Группировка фиксаций
        fixations = []
        saccades = []
        current_fixation = None
        
        # Проход по всем кадрам
        for i in range(len(df)):
            if df['is_fixation'].iloc[i]:
                # Если это фиксация
                if current_fixation is None:
                    # Начало новой фиксации
                    current_fixation = {'start_time': df['time'].iloc[i], 'x': [], 'y': []}
                # Добавление точки к текущей фиксации
                current_fixation['x'].append(df['x_smooth'].iloc[i])
                current_fixation['y'].append(df['y_smooth'].iloc[i])
            else:
                # Если это не фиксация (саккада)
                if current_fixation:
                    # Конец фиксации - проверка минимальной длительности
                    duration = df['time'].iloc[i] - current_fixation['start_time']
                    if duration >= self.min_fixation_duration:
                        # Сохранение фиксации
                        fixations.append({
                            'start_time': current_fixation['start_time'],
                            'end_time': df['time'].iloc[i],
                            'duration': duration,
                            'x_center': np.mean(current_fixation['x']),  # Центр фиксации по X
                            'y_center': np.mean(current_fixation['y']),  # Центр фиксации по Y
                            'amplitude': 0  # У фиксаций нет амплитуды
                        })
                    current_fixation = None
                
                # Если переход из фиксации в саккаду - сохраняем саккаду
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
        
        # Сохранение последней фиксации, если она не закончилась
        if current_fixation and len(df) > 0:
            duration = df['time'].iloc[-1] - current_fixation['start_time']
            if duration >= self.min_fixation_duration:
                fixations.append({
                    'start_time': current_fixation['start_time'],
                    'end_time': df['time'].iloc[-1],
                    'duration': duration,
                    'x_center': np.mean(current_fixation['x']),
                    'y_center': np.mean(current_fixation['y']),
                    'amplitude': 0
                })
        
        # Создание DataFrame для результатов
        fixations_df = pd.DataFrame(fixations)
        saccades_df = pd.DataFrame(saccades)
        
        return {'fixations': fixations_df, 'saccades': saccades_df}
