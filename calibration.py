import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PupilDetector:
    def __init__(self):
        """Инициализация детектора зрачков"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        self.trajectory_screen = []
        self.trajectory_canvas = None
        self.reduced_width = None
        self.reduced_height = None
        self.pupil_data = []
        self.calibration_factor = None  # Хранит калибровочный коэффициент

    def detect_pupils_in_frame(self, frame, frame_id=None):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)

        if self.reduced_width is None:
            h, w = frame.shape[:2]
            self.reduced_width = w // 2
            self.reduced_height = h // 2
            self.trajectory_canvas = np.zeros((self.reduced_height, self.reduced_width, 3), dtype=np.uint8)

        frame_with_pupils = cv2.resize(frame, (self.reduced_width, self.reduced_height))

        pupil_coords = {'left': None, 'right': None}

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                left_eye_idx = 468
                right_eye_idx = 473

                h, w, _ = frame.shape
                left_eye = face_landmarks.landmark[left_eye_idx]
                right_eye = face_landmarks.landmark[right_eye_idx]

                left_eye_coords = (int(left_eye.x * self.reduced_width), int(left_eye.y * self.reduced_height))
                right_eye_coords = (int(right_eye.x * self.reduced_width), int(right_eye.y * self.reduced_height))

                cv2.circle(frame_with_pupils, left_eye_coords, 5, (0, 255, 0), -1)
                cv2.circle(frame_with_pupils, right_eye_coords, 5, (0, 0, 255), -1)

                pupil_coords['left'] = left_eye_coords
                pupil_coords['right'] = right_eye_coords

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
        gaze_x = (left_eye.x + right_eye.x) / 2
        gaze_y = (left_eye.y + right_eye.y) / 2

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
        if not self.pupil_data:
            print("Нет данных о зрачках для сохранения.")
            return

        df = pd.DataFrame(self.pupil_data)
        df.to_csv(output_file, index=False)
        print(f"Данные о зрачках сохранены в {output_file}")

    def process_video(self, video_path, output_csv="pupil_data.csv", show_result=True):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = int(1000 / fps) if fps > 0 else 1

        frame_count = 0
        self.pupil_data = []

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
        input_path = Path(input_folder)
        output_folder = f"{input_path.name}_res"
        output_path = input_path.parent / output_folder
        output_path.mkdir(exist_ok=True)

        self.pupil_data = []

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
                            left_eye_idx = 468
                            right_eye_idx = 473

                            left_eye = face_landmarks.landmark[left_eye_idx]
                            right_eye = face_landmarks.landmark[right_eye_idx]

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

    def calculate_calibration(self, L_mm=177, D_mm=380, x1=None, y1=None, x2=None, y2=None):
        """
        Вычисляет калибровочный коэффициент на основе двух точек.
        :param L_mm: Расстояние между точками на листе (мм).
        :param D_mm: Расстояние от глаз до камеры (мм).
        :param x1, y1: Координаты первой точки (пиксели).
        :param x2, y2: Координаты второй точки (пиксели).
        """
        if x1 is None or y1 is None or x2 is None or y2 is None:
            print("Координаты для калибровки не указаны. Используйте данные из pupil_data.")
            return None

        theta_rad = 2 * np.arctan(L_mm / (2 * D_mm))
        theta_deg = np.degrees(theta_rad)
        delta_px = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        calibration_factor = delta_px / theta_deg
        self.calibration_factor = calibration_factor
        print(f"Калибровочный коэффициент: {calibration_factor:.2f} px/degree")
        return calibration_factor

    def calibrate_eye_movement(self, input_csv="pupil_data.csv"):
        """
        Преобразует координаты зрачков в углы с использованием калибровочного коэффициента.
        """
        try:
            df = pd.read_csv(input_csv)
        except FileNotFoundError:
            print(f"Файл {input_csv} не найден.")
            return None, None

        avg_x = (df["left_x"] + df["right_x"]) / 2
        avg_y = (df["left_y"] + df["right_y"]) / 2

        if self.calibration_factor is None:
            # Пример калибровки (замените на реальные x1, y1, x2, y2 из эксперимента)
            x1, y1 = avg_x.iloc[0], avg_y.iloc[0]
            x2, y2 = avg_x.iloc[-1], avg_y.iloc[-1]
            self.calculate_calibration(177, 380, x1, y1, x2, y2)

        if self.calibration_factor is None:
            print("Калибровочный коэффициент не вычислен.")
            return None, None

        # Преобразование пикселей в углы (относительно первой точки как базовой)
        theta_x = (avg_x - avg_x.iloc[0]) / self.calibration_factor
        theta_y = (avg_y - avg_y.iloc[0]) / self.calibration_factor

        return theta_x, theta_y

    def plot_eye_movement(self, input_csv="pupil_data.csv", output_plot="eye_movement_plot.html"):
        """
        Строит интерактивный график движения глаз с углами и производными.
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

        # Калибровка и преобразование в углы
        theta_x, theta_y = self.calibrate_eye_movement(input_csv)

        if theta_x is None or theta_y is None:
            return

        # Производные (скорость изменения углов)
        d_theta_x = np.gradient(theta_x, np.diff(time_axis, prepend=0))
        d_theta_y = np.gradient(theta_y, np.diff(time_axis, prepend=0))

        # Определение направлений для x
        dx = np.gradient(avg_x)
        directions = ["Right" if d > 0 else "Left" if d < 0 else "Still" for d in dx]

        # Создаем подграфики (4 ряда: x, dx, y, dy)
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=("Horizontal Position (px)", "Horizontal Velocity (deg/s)",
                           "Vertical Position (px)", "Vertical Velocity (deg/s)"),
            shared_xaxes=True,
            vertical_spacing=0.1
        )

        # График горизонтальной позиции (пиксели)
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=avg_x,
                mode='lines+markers',
                name='Horizontal (px)',
                line=dict(color='blue'),
                hovertemplate='Time: %{x:.3f}s<br>X: %{y:.2f} px<br>Direction: %{text}',
                text=directions
            ),
            row=1, col=1
        )

        # График скорости по горизонтали (производная)
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=d_theta_x,
                mode='lines',
                name='Horizontal Velocity (deg/s)',
                line=dict(color='cyan')
            ),
            row=2, col=1
        )

        # График вертикальной позиции (пиксели)
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=avg_y,
                mode='lines+markers',
                name='Vertical (px)',
                line=dict(color='orange'),
                hovertemplate='Time: %{x:.3f}s<br>Y: %{y:.2f} px'
            ),
            row=3, col=1
        )

        # График скорости по вертикали (производная)
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=d_theta_y,
                mode='lines',
                name='Vertical Velocity (deg/s)',
                line=dict(color='magenta')
            ),
            row=4, col=1
        )

        # Калиброванные углы
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=theta_x,
                mode='lines',
                name='Horizontal Angle (deg)',
                line=dict(color='green', dash='dash'),
                hovertemplate='Time: %{x:.3f}s<br>θ_x: %{y:.2f}°'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=theta_y,
                mode='lines',
                name='Vertical Angle (deg)',
                line=dict(color='red', dash='dash'),
                hovertemplate='Time: %{x:.3f}s<br>θ_y: %{y:.2f}°'
            ),
            row=3, col=1
        )

        # Настройка осей
        fig.update_xaxes(title_text="Time (s)", row=4, col=1)
        fig.update_yaxes(title_text="Position (pixels)", row=1, col=1)
        fig.update_yaxes(title_text="Velocity (deg/s)", row=2, col=1)
        fig.update_yaxes(title_text="Position (pixels)", row=3, col=1)
        fig.update_yaxes(title_text="Velocity (deg/s)", row=4, col=1)

        fig.update_layout(
            title="Eye Movement Analysis with Angles and Velocities",
            showlegend=True,
            height=800,
            width=800,
            hovermode="closest"
        )

        fig.write_html(output_plot)
        print(f"Интерактивный график сохранен в {output_plot}")

if __name__ == "__main__":
    detector = PupilDetector()
    detector.process_video('test_data/video_7.MOV', output_csv='pupil_data_7.csv')
    detector.plot_eye_movement(input_csv='pupil_data_7.csv', output_plot='eye_movement_plot_7.html')