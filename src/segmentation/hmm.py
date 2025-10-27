"""
Алгоритм HMM (Hidden Markov Model) для сегментации движений глаз

Использует модель скрытых марковских цепей для классификации движений глаз
на фиксации и саккады на основе статистического анализа.
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from hmmlearn import hmm
from typing import Dict


class HMMSegmenter:
    """
    Сегментатор HMM для классификации движений глаз.
    
    Использует скрытую марковскую модель для классификации движений глаз.
    Обычно определяет состояния с низкой скоростью как фиксации.
    """
    
    def __init__(self, n_components: int = 2, covariance_type: str = 'diag', 
                 n_iter: int = 100, min_fixation_duration: float = 0.1, fps: float = 30):
        """
        Инициализация сегментатора HMM.
        
        Args:
            n_components: Количество скрытых состояний (обычно 2: фиксация/саккада)
            covariance_type: Тип ковариационной матрицы ('diag' - диагональная)
            n_iter: Количество итераций обучения модели
            min_fixation_duration: Минимальная длительность фиксации в секундах
            fps: Частота кадров
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.min_fixation_duration = min_fixation_duration
        self.fps = fps
        self.model = None  # Обученная модель HMM
    
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
    
    def compute_velocity(self, df: pd.DataFrame) -> pd.Series:
        """
        Вычисление скорости движения глаз из данных о позиции.
        
        Вычисляет скорость как производную от позиции по времени.
        
        Args:
            df: DataFrame с колонками 'x_smooth', 'y_smooth', 'time'
            
        Returns:
            Series со значениями скорости
        """
        # Интервал времени между кадрами
        dt = df['time'].diff().fillna(1 / self.fps)
        # Изменение по X
        dx = df['x_smooth'].diff().fillna(0)
        # Изменение по Y
        dy = df['y_smooth'].diff().fillna(0)
        # Вычисление скорости
        velocity = np.sqrt(dx**2 + dy**2) / dt
        return velocity
    
    def segment(self, input_csv: str, output_prefix: str = "eye_movement") -> Dict[str, pd.DataFrame]:
        """
        Сегментация движений глаз с помощью HMM.
        
        Метод загружает данные о зрачках, создает модель HMM и обучает ее,
        затем использует обученную модель для классификации движений глаз.
        
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
        df['x_smooth'] = savgol_filter(df['x'], window_length=5, polyorder=2)
        df['y_smooth'] = savgol_filter(df['y'], window_length=5, polyorder=2)
        
        # Вычисление скорости
        velocity = self.compute_velocity(df)
        
        # Подготовка данных для HMM: скорость, x, y
        X = np.column_stack([velocity, df['x_smooth'], df['y_smooth']])
        # Удаление строк с NaN значениями
        X = X[~np.isnan(X).any(axis=1)]
        
        if len(X) == 0:
            return {'fixations': pd.DataFrame(), 'saccades': pd.DataFrame()}
        
        # Создание и обучение модели HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter
        )
        self.model.fit(X)
        
        # Предсказание состояний для всех точек
        states = self.model.predict(X)
        
        # Определение состояния фиксации (низкая средняя скорость)
        mean_velocity = [np.mean(X[states == i, 0]) for i in range(self.n_components)]
        fixation_state = np.argmin(mean_velocity)
        
        # Группировка фиксаций
        fixations = []
        saccades = []
        current_fixation = None
        
        for i in range(len(df)):
            # Индекс состояния в массиве состояний
            state_idx = i if i < len(states) else len(states) - 1
            # Проверка, является ли текущее состояние фиксацией
            is_fixation = states[state_idx] == fixation_state if state_idx < len(states) else False
            
            if is_fixation:
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
                    # Конец фиксации
                    duration = df['time'].iloc[i] - current_fixation['start_time']
                    if duration >= self.min_fixation_duration:
                        # Сохранение фиксации
                        fixations.append({
                            'start_time': current_fixation['start_time'],
                            'end_time': df['time'].iloc[i],
                            'duration': duration,
                            'x_center': np.mean(current_fixation['x']),
                            'y_center': np.mean(current_fixation['y']),
                            'amplitude': 0
                        })
                    current_fixation = None
                
                # Сохранение саккады при переходе из фиксации
                if i > 0 and state_idx > 0 and states[min(state_idx - 1, len(states) - 1)] == fixation_state:
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
