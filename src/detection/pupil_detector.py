"""
Модуль детектирования зрачков

Предоставляет классы для детектирования зрачков в видеокадрах и изображениях,
с опциональной компенсацией движения головы.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List


class PupilDetector:
    """
    Базовый детектор зрачков, использующий MediaPipe Face Mesh.
    
    Этот класс предназначен для детектирования зрачков на изображениях и видео.
    Использует MediaPipe для определения координат зрачков и сохраняет результаты.
    """
    
    # Индексы ориентиров MediaPipe для зрачков
    LEFT_PUPIL_IDX = 468  # Индекс левого зрачка в модели MediaPipe
    RIGHT_PUPIL_IDX = 473  # Индекс правого зрачка в модели MediaPipe
    
    def __init__(self):
        """
        Инициализация детектора зрачков.
        
        Создает экземпляр MediaPipe Face Mesh с уточненными ориентирами лиц.
        Инициализирует список для хранения данных о зрачках.
        """
        self.mp_face_mesh = mp.solutions.face_mesh  # Модуль Face Mesh из MediaPipe
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,  # Использовать уточненные ориентиры
            max_num_faces=1  # Максимум 1 лицо в кадре
        )
        self.pupil_data = []  # Список для хранения данных о зрачках
        
    def detect_pupils_in_frame(self, frame: np.ndarray, frame_id: Optional[str] = None) -> Tuple[Dict[str, Optional[Tuple[float, float]]]]:
        """
        Детектирование зрачков в одном кадре.
        
        Этот метод обрабатывает один кадр изображения и определяет координаты
        левого и правого зрачков. Координаты возвращаются в нормализованном формате [0, 1]
        относительно размеров изображения.
        
        Args:
            frame: Входной кадр как numpy массив в формате BGR (Blue-Green-Red)
            frame_id: Необязательный идентификатор кадра (например, имя файла)
            
        Returns:
            Словарь с координатами 'left' и 'right' зрачков (x, y),
            или None если зрачки не обнаружены. Координаты нормализованы [0, 1].
        """
        # Конвертация из BGR в RGB для MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)
        
        # Инициализация словаря для координат
        pupil_coords = {'left': None, 'right': None}
        
        # Если найдено лицо в кадре
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                # Получение координат зрачков из ориентиров MediaPipe
                left_eye = face_landmarks.landmark[self.LEFT_PUPIL_IDX]
                right_eye = face_landmarks.landmark[self.RIGHT_PUPIL_IDX]
                
                # Сохранение нормализованных координат (0-1)
                pupil_coords['left'] = (left_eye.x, left_eye.y)
                pupil_coords['right'] = (right_eye.x, right_eye.y)
                
                # Сохранение координат в пикселях (оригинальное разрешение)
                h, w = frame.shape[:2]  # Получение высоты и ширины кадра
                self.pupil_data.append({
                    "filename": frame_id if frame_id else f"frame_{len(self.pupil_data)}",
                    "left_x": int(left_eye.x * w),  # X координата левого зрачка в пикселях
                    "left_y": int(left_eye.y * h),  # Y координата левого зрачка в пикселях
                    "right_x": int(right_eye.x * w),  # X координата правого зрачка в пикселях
                    "right_y": int(right_eye.y * h),  # Y координата правого зрачка в пикселях
                    "left_x_norm": left_eye.x,  # Нормализованная X координата левого зрачка
                    "left_y_norm": left_eye.y,  # Нормализованная Y координата левого зрачка
                    "right_x_norm": right_eye.x,  # Нормализованная X координата правого зрачка
                    "right_y_norm": right_eye.y  # Нормализованная Y координата правого зрачка
                })
        
        return pupil_coords
    
    def process_video(self, video_path, output_csv: str = "pupil_data.csv", show_result: bool = False):
        """
        Обработка видеофайла и детектирование зрачков.
        
        Этот метод обрабатывает весь видеофайл, кадр за кадром, детектируя
        координаты зрачков в каждом кадре. Результаты сохраняются в CSV файл.
        
        Args:
            video_path: Путь к видеофайлу или 0 для веб-камеры
            output_csv: Путь к CSV файлу для сохранения координат зрачков
            show_result: Показывать ли видео в реальном времени (по умолчанию False)
        """
        # Открытие видеофайла
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
            
        # Получение FPS (кадров в секунду) видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Задержка между кадрами для воспроизведения (миллисекунды)
        frame_delay = int(1000 / fps) if fps > 0 else 1
        
        frame_count = 0
        self.pupil_data = []  # Очистка списка данных
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps:.2f}")
        
        # Обработка каждого кадра
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Вычисление времени кадра для создания идентификатора
            frame_time = frame_count / fps
            seconds = int(frame_time)
            microseconds = int((frame_time - seconds) * 1_000_000)
            # Формат: frame_0_00_SS_MMMMMM.jpg
            frame_id = f"frame_0_00_{seconds:02d}_{microseconds:06d}.jpg"
            
            # Детектирование зрачков в текущем кадре
            self.detect_pupils_in_frame(frame, frame_id)
            frame_count += 1
            
            # Периодический вывод прогресса
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        # Освобождение ресурсов
        cap.release()
        # Сохранение результатов
        self.save_pupil_data(output_csv)
        print(f"Saved pupil data to {output_csv}")
    
    def process_image_folder(self, input_folder: str, output_csv: str = "pupil_data.csv"):
        """
        Обработка всех изображений в папке и детектирование зрачков.
        
        Этот метод обрабатывает все изображения в указанной папке,
        детектируя координаты зрачков на каждом изображении.
        
        Args:
            input_folder: Путь к папке, содержащей изображения
            output_csv: Путь к CSV файлу для сохранения координат зрачков
        """
        input_path = Path(input_folder)
        if not input_path.exists():
            raise ValueError(f"Folder not found: {input_folder}")
        
        self.pupil_data = []  # Очистка списка данных
        
        # Поиск всех JPEG и PNG изображений
        image_files = sorted(input_path.glob('*.jpg')) + sorted(input_path.glob('*.png'))
        total = len(image_files)
        
        print(f"Processing {total} images from {input_folder}")
        
        # Обработка каждого изображения
        for i, img_file in enumerate(image_files):
            try:
                # Загрузка изображения
                frame = cv2.imread(str(img_file))
                if frame is None:
                    print(f"Error: Could not read image {img_file.name}")
                    continue
                
                # Детектирование зрачков на изображении
                self.detect_pupils_in_frame(frame, img_file.name)
                
                # Периодический вывод прогресса (каждые 10 изображений)
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{total} images")
                    
            except Exception as e:
                print(f"Error processing {img_file.name}: {str(e)}")
        
        # Сохранение результатов
        self.save_pupil_data(output_csv)
        print(f"Saved pupil data to {output_csv}")
    
    def save_pupil_data(self, output_file: str = "pupil_data.csv"):
        """
        Сохранение данных о зрачках в CSV файл.
        
        Метод создает DataFrame из собранных данных о зрачках и сохраняет
        его в CSV файл для дальнейшего анализа и обработки.
        
        Args:
            output_file: Путь к CSV файлу для сохранения
        """
        if not self.pupil_data:
            print("No pupil data to save.")
            return
        
        # Создание DataFrame и сохранение в CSV
        df = pd.DataFrame(self.pupil_data)
        df.to_csv(output_file, index=False)
        print(f"Pupil data saved to {output_file} ({len(df)} frames)")


class PupilHeadDetector(PupilDetector):
    """
    Детектор зрачков с компенсацией движения головы.
    
    Этот класс наследуется от PupilDetector и добавляет функциональность
    компенсации движения головы для более точного отслеживания движений глаз.
    Использует референсные точки лица (переносица, уголки глаз) для определения
    и компенсации движения головы.
    """
    
    # Индексы референсных точек головы в MediaPipe
    NOSE_BRIDGE_IDX = 168  # Переносица
    LEFT_EYE_OUTER_IDX = 33  # Левый внешний угол глаза
    RIGHT_EYE_OUTER_IDX = 263  # Правый внешний угол глаза
    
    def __init__(self, head_movement_threshold: float = 0.01):
        """
        Инициализация детектора с компенсацией движения головы.
        
        Args:
            head_movement_threshold: Минимальное движение для учета движения головы.
                                     Значения меньше этого порога игнорируются.
        """
        super().__init__()  # Вызов конструктора родительского класса
        self.prev_head_position = None  # Предыдущее положение головы
        self.head_movement_threshold = head_movement_threshold  # Порог движения головы
    
    def _get_head_reference_points(self, face_landmarks) -> Dict[str, Tuple[float, float]]:
        """
        Получение референсных точек для определения движения головы.
        
        Использует переносицу и внешние уголки глаз для создания
        системы отсчета для компенсации движения головы.
        
        Args:
            face_landmarks: Ориентиры лица из MediaPipe
            
        Returns:
            Словарь с координатами референсных точек: nose_bridge, left_eye_outer, right_eye_outer
        """
        # Получение координат переносицы
        nose_bridge = face_landmarks.landmark[self.NOSE_BRIDGE_IDX]
        # Получение координат левого внешнего угла глаза
        left_eye_outer = face_landmarks.landmark[self.LEFT_EYE_OUTER_IDX]
        # Получение координат правого внешнего угла глаза
        right_eye_outer = face_landmarks.landmark[self.RIGHT_EYE_OUTER_IDX]
        
        return {
            'nose_bridge': (nose_bridge.x, nose_bridge.y),  # Переносица
            'left_eye_outer': (left_eye_outer.x, left_eye_outer.y),  # Левый угол глаза
            'right_eye_outer': (right_eye_outer.x, right_eye_outer.y)  # Правый угол глаза
        }
    
    def _compute_head_movement(self, current_head_pos: Dict) -> Tuple[float, float]:
        """
        Вычисление движения головы относительно предыдущего положения.
        
        Сравнивает текущее положение референсных точек с предыдущим
        для определения движения головы в нормализованных координатах.
        
        Args:
            current_head_pos: Текущее положение референсных точек
            
        Returns:
            Кортеж (dx, dy) - смещение головы по X и Y оси
        """
        # Для первого кадра предыдущее положение отсутствует
        if self.prev_head_position is None:
            self.prev_head_position = current_head_pos
            return (0.0, 0.0)  # Нет движения
        
        # Среднее изменение положения по всем референсным точкам по X
        dx = np.mean([
            current_head_pos['nose_bridge'][0] - self.prev_head_position['nose_bridge'][0],
            current_head_pos['left_eye_outer'][0] - self.prev_head_position['left_eye_outer'][0],
            current_head_pos['right_eye_outer'][0] - self.prev_head_position['right_eye_outer'][0]
        ])
        
        # Среднее изменение положения по всем референсным точкам по Y
        dy = np.mean([
            current_head_pos['nose_bridge'][1] - self.prev_head_position['nose_bridge'][1],
            current_head_pos['left_eye_outer'][1] - self.prev_head_position['left_eye_outer'][1],
            current_head_pos['right_eye_outer'][1] - self.prev_head_position['right_eye_outer'][1]
        ])
        
        # Обновление предыдущего положения
        self.prev_head_position = current_head_pos
        
        # Игнорирование очень маленьких движений (ниже порога)
        if abs(dx) < self.head_movement_threshold and abs(dy) < self.head_movement_threshold:
            return (0.0, 0.0)
        
        return (dx, dy)
    
    def _get_relative_pupil_position(self, pupil_coords: Tuple[float, float], 
                                    head_ref_points: Dict) -> Tuple[float, float]:
        """
        Вычисление относительного положения зрачка относительно референсных точек головы.
        
        Использует переносицу как точку отсчета для вычисления относительного
        положения зрачка, компенсируя движения головы.
        
        Args:
            pupil_coords: Координаты зрачка (x, y)
            head_ref_points: Референсные точки головы
            
        Returns:
            Кортеж относительных координат зрачка
        """
        ref_x, ref_y = head_ref_points['nose_bridge']  # Получение координат переносицы
        # Относительные координаты зрачка относительно переносицы
        return (pupil_coords[0] - ref_x, pupil_coords[1] - ref_y)
    
    def detect_pupils_in_frame(self, frame: np.ndarray, frame_id: Optional[str] = None) -> Tuple[Dict, Dict]:
        """
        Детектирование зрачков с компенсацией движения головы.
        
        Этот метод определяет координаты зрачков и вычисляет их относительное
        положение с компенсацией движения головы. Возвращает как абсолютные,
        так и относительные (с компенсацией) координаты.
        
        Args:
            frame: Входной кадр как numpy массив в формате BGR
            frame_id: Необязательный идентификатор кадра
            
        Returns:
            Кортеж из двух словарей: (absolute_coords, relative_coords)
            - absolute_coords: абсолютные координаты зрачков
            - relative_coords: относительные координаты с компенсацией движения головы
        """
        # Конвертация в RGB для MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)
        
        h, w = frame.shape[:2]
        pupil_coords = {'left': None, 'right': None}
        relative_pupil_coords = {'left': None, 'right': None}
        
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                # Получение референсных точек головы
                head_ref_points = self._get_head_reference_points(face_landmarks)
                # Вычисление движения головы
                head_movement = self._compute_head_movement(head_ref_points)
                
                # Получение координат зрачков
                left_eye = face_landmarks.landmark[self.LEFT_PUPIL_IDX]
                right_eye = face_landmarks.landmark[self.RIGHT_PUPIL_IDX]
                
                # Абсолютные координаты в пикселях
                pupil_coords['left'] = (int(left_eye.x * w), int(left_eye.y * h))
                pupil_coords['right'] = (int(right_eye.x * w), int(right_eye.y * h))
                
                # Вычисление относительных координат
                relative_left = self._get_relative_pupil_position((left_eye.x, left_eye.y), head_ref_points)
                relative_right = self._get_relative_pupil_position((right_eye.x, right_eye.y), head_ref_points)
                
                # Компенсация движения головы
                compensated_left = (relative_left[0] - head_movement[0], relative_left[1] - head_movement[1])
                compensated_right = (relative_right[0] - head_movement[0], relative_right[1] - head_movement[1])
                
                relative_pupil_coords['left'] = compensated_left
                relative_pupil_coords['right'] = compensated_right
                
                # Сохранение данных (абсолютные и относительные координаты)
                self.pupil_data.append({
                    "filename": frame_id if frame_id else f"frame_{len(self.pupil_data)}",
                    "left_x": pupil_coords['left'][0],
                    "left_y": pupil_coords['left'][1],
                    "right_x": pupil_coords['right'][0],
                    "right_y": pupil_coords['right'][1],
                    "rel_left_x": compensated_left[0],  # Относительная X координата левого зрачка
                    "rel_left_y": compensated_left[1],  # Относительная Y координата левого зрачка
                    "rel_right_x": compensated_right[0],  # Относительная X координата правого зрачка
                    "rel_right_y": compensated_right[1]  # Относительная Y координата правого зрачка
                })
        
        return pupil_coords, relative_pupil_coords
