import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

from eye_tracker.data_processing import smooth_coordinates, compute_velocity
from eye_tracker.utils import load_pupil_data


class PupilDetector:
    def __init__(self):
        """Инициализация детектора зрачков"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        self.trajectory_screen = []
        self.trajectory_canvas = None
        self.reduced_width = None
        self.reduced_height = None
        self.pupil_data = []  # Для хранения координат зрачков

    def detect_pupils_in_frame(self, frame, frame_id=None):
        """
        Детекция зрачков в одном кадре
        :param frame: входной кадр (numpy array)
        :param frame_id: идентификатор кадра (например, имя файла или номер)
        :return: кадр с отмеченными зрачками, координаты зрачков
        """
        # Исходное разрешение кадра
        h, w = frame.shape[:2]

        # Инициализация уменьшенного разрешения для визуализации
        if self.reduced_width is None:
            self.reduced_width = w // 2
            self.reduced_height = h // 2
            self.trajectory_canvas = np.zeros((self.reduced_height, self.reduced_width, 3), dtype=np.uint8)

        # Преобразование кадра для Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)

        # Уменьшенный кадр только для визуализации
        frame_with_pupils = cv2.resize(frame, (self.reduced_width, self.reduced_height))

        pupil_coords = {'left': None, 'right': None}

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                left_eye_idx = 468  # Левый зрачок
                right_eye_idx = 473  # Правый зрачок

                left_eye = face_landmarks.landmark[left_eye_idx]
                right_eye = face_landmarks.landmark[right_eye_idx]

                # Координаты в оригинальном разрешении (масштабируем к w, h)
                left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
                right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))

                # Координаты для визуализации (масштабируем к уменьшенному разрешению)
                left_eye_coords_reduced = (int(left_eye.x * self.reduced_width), int(left_eye.y * self.reduced_height))
                right_eye_coords_reduced = (int(right_eye.x * self.reduced_width), int(right_eye.y * self.reduced_height))

                # Отрисовка зрачков на уменьшенном кадре
                cv2.circle(frame_with_pupils, left_eye_coords_reduced, 5, (0, 255, 0), -1)
                cv2.circle(frame_with_pupils, right_eye_coords_reduced, 5, (0, 0, 255), -1)

                pupil_coords['left'] = left_eye_coords
                pupil_coords['right'] = right_eye_coords

                # Сохранение координат в оригинальном разрешении
                self.pupil_data.append({
                    "filename": frame_id if frame_id else f"frame_{len(self.pupil_data)}",
                    "left_x": left_eye_coords[0],
                    "left_y": left_eye_coords[1],
                    "right_x": right_eye_coords[0],
                    "right_y": right_eye_coords[1]
                })

                self._update_gaze_trajectory(left_eye, right_eye)

        combined_frame = np.hstack((frame_with_pupils, self.trajectory_canvas))
        return combined_frame, pupil_coords

    def _update_gaze_trajectory(self, left_eye, right_eye):
        """Обновление траектории взгляда (в уменьшенном разрешении)"""
        gaze_x = (left_eye.x + right_eye.x) / 2
        gaze_y = (left_eye.y + right_eye.y) / 2

        # Масштабируем к уменьшенному разрешению для траектории
        screen_x = int(gaze_x * self.reduced_width)
        screen_y = int(gaze_y * self.reduced_height)

        self.trajectory_screen.append((screen_x, screen_y))

        if len(self.trajectory_screen) > 100:
            self.trajectory_screen.pop(0)

        for i in range(1, len(self.trajectory_screen)):
            cv2.line(self.trajectory_canvas, self.trajectory_screen[i - 1],
                     self.trajectory_screen[i], (255, 255, 255), 2)

        cv2.circle(self.trajectory_canvas, (screen_x, screen_y), 2, (0, 255, 0), -1)

    def save_pupil_data(self, output_file="pupil_data.csv"):
        """Сохраняет данные о координатах зрачков в CSV-файл."""
        if not self.pupil_data:
            print("Нет данных о зрачках для сохранения.")
            return

        df = pd.DataFrame(self.pupil_data)
        df.to_csv(output_file, index=False)
        print(f"Данные о зрачках сохранены в {output_file}")

    def process_video(self, video_path, output_csv="pupil_data.csv", show_result=True):
        """
        Обработка видеофайла или потока с камеры
        :param video_path: путь к видео или 0 для веб-камеры
        :param output_csv: путь к CSV-файлу для сохранения координат
        :param show_result: показывать ли результат в реальном времени
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = int(1000 / fps) if fps > 0 else 1

        frame_count = 0
        self.pupil_data = []  # Очищаем данные перед обработкой

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_time = frame_count / fps
            seconds = int(frame_time)
            microseconds = int((frame_time - seconds) * 1_000_000)
            frame_id = f"frame_0_00_{seconds:02d}_{microseconds:06d}.jpg"

            result_frame, _ = self.detect_pupils_in_frame(frame, frame_id)
            frame_count += 1

            if show_result:
                cv2.imshow("Video and Eye Trajectory", result_frame)

            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        self.save_pupil_data(output_csv)

    def process_image_folder(self, input_folder, output_csv="pupil_data.csv", circle_radius=1):
        """
        Обработка всех изображений в папке - выделение зрачков на фото
        :param input_folder: путь к папке с изображениями
        :param output_csv: путь к CSV-файлу для сохранения координат
        :param circle_radius: радиус кругов для выделения зрачков
        """
        input_path = Path(input_folder)
        output_folder = f"{input_path.name}_res"
        output_path = input_path.parent / output_folder
        output_path.mkdir(exist_ok=True)

        self.pupil_data = []  # Очищаем данные перед обработкой

        for img_file in input_path.glob('*.*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                try:
                    frame = cv2.imread(str(img_file))
                    if frame is None:
                        print(f"Error: Could not read image {img_file.name}")
                        continue

                    h, w = frame.shape[:2]

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.face_mesh.process(rgb_frame)

                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            left_eye_idx = 468  # Левый зрачок
                            right_eye_idx = 473  # Правый зрачок

                            left_eye = face_landmarks.landmark[left_eye_idx]
                            right_eye = face_landmarks.landmark[right_eye_idx]

                            # Координаты в оригинальном разрешении
                            original_coords_left = (int(left_eye.x * w), int(left_eye.y * h))
                            original_coords_right = (int(right_eye.x * w), int(right_eye.y * h))

                            cv2.circle(frame, original_coords_left, circle_radius, (0, 255, 0), -1)
                            cv2.circle(frame, original_coords_right, circle_radius, (0, 0, 255), -1)

                            self.pupil_data.append({
                                "filename": img_file.name,
                                "left_x": original_coords_left[0],
                                "left_y": original_coords_left[1],
                                "right_x": original_coords_right[0],
                                "right_y": original_coords_right[1]
                            })

                            output_file = output_path / img_file.name
                            cv2.imwrite(str(output_file), frame)
                            print(f"Successfully processed: {img_file.name}")

                except Exception as e:
                    print(f"Error processing {img_file.name}: {str(e)}")

        self.save_pupil_data(output_csv)

    def plot_eye_movement(self, input_csv="pupil_data.csv", output_plot="eye_movement_plot.png"):
        """
        Строит график движения глаз по горизонтали (x) и вертикали (y)
        :param input_csv: путь к CSV-файлу с координатами зрачков
        :param output_plot: путь для сохранения графика
        """
        try:
            df = pd.read_csv(input_csv)
        except FileNotFoundError:
            print(f"Файл {input_csv} не найден.")
            return

        avg_x = (df["left_x"] + df["right_x"]) / 2
        avg_y = (df["left_y"] + df["right_y"]) / 2

        if "filename" in df.columns:
            times = []
            for fname in df["filename"]:
                parts = fname.split('_')
                if len(parts) >= 5:
                    seconds = int(parts[3])
                    microseconds = int(parts[4].split('.')[0])
                    times.append(seconds + microseconds / 1_000_000)
                else:
                    times.append(len(times))
            time_axis = times
        else:
            time_axis = range(len(avg_x))

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, avg_x, label="Horizontal (x)", color="blue")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (pixels)")
        plt.title("Eye Movement: Horizontal")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(time_axis, avg_y, label="Vertical (y)", color="orange")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (pixels)")
        plt.title("Eye Movement: Vertical")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_plot)
        plt.close()
        print(f"График сохранен в {output_plot}")

    def plot_eye_movement_ploty(self, input_csv="pupil_data.csv", output_plot="eye_movement_plot.html"):
        """
        Строит интерактивный график движения глаз по горизонтали (x) и вертикали (y)
        :param input_csv: путь к CSV-файлу с координатами зрачков
        :param output_plot: путь для сохранения графика (HTML-файл)
        """
        try:
            df = pd.read_csv(input_csv)
        except FileNotFoundError:
            print(f"Файл {input_csv} не найден.")
            return

        avg_x = (df["left_x"] + df["right_x"]) / 2
        avg_y = (df["left_y"] + df["right_y"]) / 2

        if "filename" in df.columns:
            times = []
            for fname in df["filename"]:
                parts = fname.split('_')
                if len(parts) >= 5:
                    seconds = int(parts[3])
                    microseconds = int(parts[4].split('.')[0])
                    times.append(seconds + microseconds / 1_000_000)
                else:
                    times.append(len(times))
            time_axis = times
        else:
            time_axis = range(len(avg_x))

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Eye Movement: Horizontal (x)", "Eye Movement: Vertical (y)"),
            shared_xaxes=True,
            vertical_spacing=0.1
        )

        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=avg_x,
                mode='lines+markers',
                name='Horizontal (x)',
                line=dict(color='blue'),
                hovertemplate='Time: %{x:.3f}s<br>X: %{y:.2f} px'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=avg_y,
                mode='lines+markers',
                name='Vertical (y)',
                line=dict(color='orange'),
                hovertemplate='Time: %{x:.3f}s<br>Y: %{y:.2f} px'
            ),
            row=2, col=1
        )

        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Position (pixels)", row=1, col=1)
        fig.update_yaxes(title_text="Position (pixels)", row=2, col=1)

        fig.update_layout(
            title="Eye Movement Analysis",
            showlegend=True,
            height=600,
            width=800,
            hovermode="closest"
        )

        fig.write_html(output_plot)
        print(f"Интерактивный график сохранен в {output_plot}")

if __name__ == "__main__":
    detector = PupilDetector()
    #detector.process_video('test_data/video_8.MOV', output_csv='pupil_data_8.csv')
    detector.plot_eye_movement_ploty(input_csv='pupil_data_8.csv', output_plot='eye_movement_plot_8.html')