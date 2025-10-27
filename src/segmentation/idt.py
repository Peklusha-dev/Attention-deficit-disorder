"""
Алгоритм I-DT (I-Dispersion-Threshold) для сегментации движений глаз

Классифицирует движения глаз на основе дисперсии. Фиксации определяются как
периоды, когда дисперсия координат взгляда в окне ниже заданного порога.
"""

import pandas as pd
import numpy as np
from typing import Dict


class IDTSegmenter:
    """
    Сегментатор I-DT для классификации движений глаз.
    
    Классифицирует фиксации на основе пространственной дисперсии точек взгляда.
    Если дисперсия координат в окне ниже порога - это фиксация.
    """
    
    def __init__(self, dispersion_threshold: float = 7, window_duration: float = 0.2, 
                 min_fixation_duration: float = 0.1, fps: float = 30):
        """
        Инициализация сегментатора I-DT.
        
        Args:
            dispersion_threshold: Максимальная дисперсия в пикселях.
                                 Если дисперсия ниже - это фиксация.
            window_duration: Размер окна анализа в секундах.
            min_fixation_duration: Минимальная длительность фиксации в секундах.
            fps: Частота кадров
        """
        self.dispersion_threshold = dispersion_threshold
        self.window_duration = window_duration
        self.min_fixation_duration = min_fixation_duration
        self.fps = fps
        # Размер окна в количестве кадров
        self.window_size = int(window_duration * fps)
    
    @staticmethod
    def parse_time(filename: str) -> float:
        """
        Парсинг времени из имени файла.
        
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
            # Добавление микросекунд
            microseconds = int(parts[3]) if len(parts) > 3 else 0
            return seconds + microseconds / 1_000_000
        except:
            return 0.0
    
    @staticmethod
    def dispersion(x: pd.Series, y: pd.Series) -> float:
        """
        Вычисление дисперсии координат.
        
        Формула: (max(x) - min(x)) + (max(y) - min(y))
        Измеряет разброс точек в пространстве.
        
        Args:
            x: Координаты по X
            y: Координаты по Y
            
        Returns:
            Значение дисперсии в пикселях
        """
        return (np.max(x) - np.min(x)) + (np.max(y) - np.min(y))
    
    def segment(self, input_csv: str, output_prefix: str = "eye_movement") -> Dict[str, pd.DataFrame]:
        """
        Сегментация движений глаз с помощью алгоритма I-DT.
        
        Метод загружает данные о зрачках и применяет алгоритм дисперсии
        для определения фиксаций. Если дисперсия в окне ниже порога - это фиксация.
        
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
        
        # Детектирование фиксаций по скользящему окну
        fixations = []
        i = 0
        # Проход по всем кадрам
        while i < len(df) - self.window_size:
            # Текущее окно анализа
            window = df.iloc[i:i + self.window_size]
            # Проверка дисперсии в окне
            if self.dispersion(window['x'], window['y']) < self.dispersion_threshold:
                # Найдена низкая дисперсия - это начало фиксации
                start_idx = i
                start_time = window['time'].iloc[0]
                
                # Продолжение фиксации пока дисперсия низкая
                while i < len(df) - self.window_size and \
                      self.dispersion(df['x'].iloc[i:i + self.window_size],
                                     df['y'].iloc[i:i + self.window_size]) < self.dispersion_threshold:
                    i += 1
                
                # Конец фиксации
                end_idx = i + self.window_size - 1
                end_time = df['time'].iloc[end_idx]
                
                # Проверка минимальной длительности
                duration = end_time - start_time
                if duration >= self.min_fixation_duration:
                    # Сохранение фиксации
                    fixations.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'x_center': np.mean(df['x'].iloc[start_idx:end_idx + 1]),
                        'y_center': np.mean(df['y'].iloc[start_idx:end_idx + 1]),
                        'amplitude': 0
                    })
            else:
                # Высокая дисперсия - не фиксация, переходим к следующему кадру
                i += 1
        
        # Детектирование саккад между фиксациями
        saccades = []
        for j in range(1, len(fixations)):
            # Саккада - переход между двумя фиксациями
            saccades.append({
                'start_time': fixations[j - 1]['end_time'],  # Конец предыдущей фиксации
                'end_time': fixations[j]['start_time'],  # Начало следующей фиксации
                'start_x': fixations[j - 1]['x_center'],
                'start_y': fixations[j - 1]['y_center'],
                'end_x': fixations[j]['x_center'],
                'end_y': fixations[j]['y_center'],
                'amplitude': np.sqrt((fixations[j]['x_center'] - fixations[j - 1]['x_center']) ** 2 +
                                   (fixations[j]['y_center'] - fixations[j - 1]['y_center']) ** 2),
                'duration': fixations[j]['start_time'] - fixations[j - 1]['end_time']
            })
        
        # Создание DataFrame для результатов
        fixations_df = pd.DataFrame(fixations)
        saccades_df = pd.DataFrame(saccades)
        
        return {'fixations': fixations_df, 'saccades': saccades_df}
