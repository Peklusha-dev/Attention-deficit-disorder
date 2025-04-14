import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path


class PupilDetector:
	def __init__(self):
		"""Инициализация детектора зрачков"""
		self.mp_face_mesh = mp.solutions.face_mesh
		self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
		self.trajectory_screen = []
		self.trajectory_canvas = None
		self.reduced_width = None
		self.reduced_height = None

	def detect_pupils_in_frame(self, frame):
		"""
        Детекция зрачков в одном кадре
        :param frame: входной кадр (numpy array)
        :return: кадр с отмеченными зрачками, координаты зрачков
        """
		# Преобразование кадра для Mediapipe
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
				# Индексы для зрачков в Mediapipe
				left_eye_idx = 468  # Левый зрачок
				right_eye_idx = 473  # Правый зрачок

				# Получение координат зрачков
				h, w, _ = frame.shape
				left_eye = face_landmarks.landmark[left_eye_idx]
				right_eye = face_landmarks.landmark[right_eye_idx]

				left_eye_coords = (int(left_eye.x * self.reduced_width), int(left_eye.y * self.reduced_height))
				right_eye_coords = (int(right_eye.x * self.reduced_width), int(right_eye.y * self.reduced_height))

				# Отрисовка зрачков
				cv2.circle(frame_with_pupils, left_eye_coords, 5, (0, 255, 0), -1)
				cv2.circle(frame_with_pupils, right_eye_coords, 5, (0, 0, 255), -1)

				pupil_coords['left'] = left_eye_coords
				pupil_coords['right'] = right_eye_coords

				# Обновление траектории взгляда
				self._update_gaze_trajectory(left_eye, right_eye)

		# Объединение видео и траектории
		combined_frame = np.hstack((frame_with_pupils, self.trajectory_canvas))
		return combined_frame, pupil_coords

	def _update_gaze_trajectory(self, left_eye, right_eye):
		"""Обновление траектории взгляда"""
		# Средняя точка между глазами (направление взгляда)
		gaze_x = (left_eye.x + right_eye.x) / 2
		gaze_y = (left_eye.y + right_eye.y) / 2

		# Преобразование координат взгляда в экранные
		screen_x = int(gaze_x * self.reduced_width)
		screen_y = int(gaze_y * self.reduced_height)

		# Сохранение траектории взгляда
		self.trajectory_screen.append((screen_x, screen_y))

		# Ограничение длины траектории
		if len(self.trajectory_screen) > 100:
			self.trajectory_screen.pop(0)

		# Рисование траектории на экране
		for i in range(1, len(self.trajectory_screen)):
			cv2.line(self.trajectory_canvas, self.trajectory_screen[i - 1],
			         self.trajectory_screen[i], (255, 255, 255), 2)

		# Рисование текущей точки взгляда
		cv2.circle(self.trajectory_canvas, (screen_x, screen_y), 2, (0, 255, 0), -1)

	def process_video(self, video_path, show_result=True):
		"""
        Обработка видеофайла или потока с камеры
        :param video_path: путь к видео или 0 для веб-камеры
        :param show_result: показывать ли результат в реальном времени
        """
		cap = cv2.VideoCapture(video_path)
		fps = cap.get(cv2.CAP_PROP_FPS)
		frame_delay = int(1000 / fps) if fps > 0 else 1

		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				break

			result_frame, _ = self.detect_pupils_in_frame(frame)

			if show_result:
				cv2.imshow("Video and Eye Trajectory", result_frame)

			if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()

	def process_image_folder(self, input_folder, circle_radius=1):
		"""
		Обработка всех изображений в папке - выделение зрачков на фото
		:param input_folder: путь к папке с изображениями
		:param circle_radius: радиус кругов для выделения зрачков (по умолчанию 10)
		"""
		input_path = Path(input_folder)
		output_folder = f"{input_path.name}_res"
		output_path = input_path.parent / output_folder
		output_path.mkdir(exist_ok=True)

		for img_file in input_path.glob('*.*'):
			if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
				try:
					frame = cv2.imread(str(img_file))
					if frame is None:
						print(f"Error: Could not read image {img_file.name}")
						continue

					h, w = frame.shape[:2]

					# Преобразование кадра для Mediapipe
					rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					results = self.face_mesh.process(rgb_frame)

					if results.multi_face_landmarks:
						for face_landmarks in results.multi_face_landmarks:
							# Индексы для зрачков
							left_eye_idx = 468  # Левый зрачок
							right_eye_idx = 473  # Правый зрачок

							# Получение координат зрачков
							left_eye = face_landmarks.landmark[left_eye_idx]
							right_eye = face_landmarks.landmark[right_eye_idx]

							# Расчет координат на оригинальном изображении
							original_coords_left = (int(left_eye.x * w), int(left_eye.y * h))
							original_coords_right = (int(right_eye.x * w), int(right_eye.y * h))

							# Отрисовка зрачков
							cv2.circle(frame, original_coords_left, circle_radius, (0, 255, 0), -1)
							cv2.circle(frame, original_coords_right, circle_radius, (0, 0, 255), -1)

							# Сохранение результата
							output_file = output_path / img_file.name
							cv2.imwrite(str(output_file), frame)
							print(f"Successfully processed: {img_file.name}")

				except Exception as e:
					print(f"Error processing {img_file.name}: {str(e)}")



if __name__ == "__main__":
	detector = PupilDetector()

# Пример использования:
# 1. Обработка видео
	#detector.process_video('test_data/video4.mp4')  # или 0 для веб-камеры

# 2. Обработка папки с изображениями
	#detector.process_image_folder('test_data/video1_frames')
