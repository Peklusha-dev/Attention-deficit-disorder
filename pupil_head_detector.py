import numpy as np
import cv2
import mediapipe as mp


class PupilHeadDetector:
	def __init__(self):
		"""Инициализация детектора зрачков"""
		self.mp_face_mesh = mp.solutions.face_mesh
		self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
		self.trajectory_screen = []
		self.trajectory_canvas = None
		self.reduced_width = None
		self.reduced_height = None
		self.prev_head_position = None
		self.head_movement_threshold = 0.01  # Порог для учета движения головы

	def _get_head_reference_points(self, face_landmarks):
		"""Получение референсных точек для головы (например, переносица и внешние уголки глаз)"""
		# Индексы точек в Mediapipe Face Mesh
		nose_bridge_idx = 168  # Переносица
		left_eye_outer_idx = 33  # Левый внешний угол глаза
		right_eye_outer_idx = 263  # Правый внешний угол глаза

		nose_bridge = face_landmarks.landmark[nose_bridge_idx]
		left_eye_outer = face_landmarks.landmark[left_eye_outer_idx]
		right_eye_outer = face_landmarks.landmark[right_eye_outer_idx]

		return {
			'nose_bridge': (nose_bridge.x, nose_bridge.y),
			'left_eye_outer': (left_eye_outer.x, left_eye_outer.y),
			'right_eye_outer': (right_eye_outer.x, right_eye_outer.y)
		}

	def _compute_head_movement(self, current_head_pos):
		"""Вычисление движения головы относительно предыдущего положения"""
		if self.prev_head_position is None:
			self.prev_head_position = current_head_pos
			return (0, 0)  # Нет движения

		# Среднее изменение положения референсных точек
		dx = np.mean([
			current_head_pos['nose_bridge'][0] - self.prev_head_position['nose_bridge'][0],
			current_head_pos['left_eye_outer'][0] - self.prev_head_position['left_eye_outer'][0],
			current_head_pos['right_eye_outer'][0] - self.prev_head_position['right_eye_outer'][0]
		])
		dy = np.mean([
			current_head_pos['nose_bridge'][1] - self.prev_head_position['nose_bridge'][1],
			current_head_pos['left_eye_outer'][1] - self.prev_head_position['left_eye_outer'][1],
			current_head_pos['right_eye_outer'][1] - self.prev_head_position['right_eye_outer'][1]
		])

		self.prev_head_position = current_head_pos

		# Игнорируем очень маленькие движения
		if abs(dx) < self.head_movement_threshold and abs(dy) < self.head_movement_threshold:
			return (0, 0)

		return (dx, dy)

	def _get_relative_pupil_position(self, pupil_coords, head_ref_points):
		"""Вычисление относительного положения зрачка относительно референсных точек"""
		# Используем переносицу как основную точку отсчета
		ref_x, ref_y = head_ref_points['nose_bridge']

		# Относительные координаты зрачка
		rel_x = pupil_coords[0] - ref_x
		rel_y = pupil_coords[1] - ref_y

		return (rel_x, rel_y)

	def detect_pupils_in_frame(self, frame):
		"""
		Детекция зрачков в одном кадре с компенсацией движения головы
		:param frame: входной кадр (numpy array)
		:return: кадр с отмеченными зрачками, координаты зрачков, относительные координаты зрачков
		"""
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		result = self.face_mesh.process(rgb_frame)

		if self.reduced_width is None:
			h, w = frame.shape[:2]
			self.reduced_width = w // 2
			self.reduced_height = h // 2
			self.trajectory_canvas = np.zeros((self.reduced_height, self.reduced_width, 3), dtype=np.uint8)

		frame_with_pupils = cv2.resize(frame, (self.reduced_width, self.reduced_height))

		pupil_coords = {'left': None, 'right': None}
		relative_pupil_coords = {'left': None, 'right': None}

		if result.multi_face_landmarks:
			for face_landmarks in result.multi_face_landmarks:
				# Получаем референсные точки головы
				head_ref_points = self._get_head_reference_points(face_landmarks)

				# Вычисляем движение головы
				head_movement = self._compute_head_movement(head_ref_points)

				# Индексы для зрачков
				left_eye_idx = 468
				right_eye_idx = 473

				# Получение координат зрачков
				left_eye = face_landmarks.landmark[left_eye_idx]
				right_eye = face_landmarks.landmark[right_eye_idx]

				left_eye_coords = (int(left_eye.x * self.reduced_width), int(left_eye.y * self.reduced_height))
				right_eye_coords = (int(right_eye.x * self.reduced_width), int(right_eye.y * self.reduced_height))

				# Вычисляем относительные координаты зрачков
				relative_left = self._get_relative_pupil_position((left_eye.x, left_eye.y), head_ref_points)
				relative_right = self._get_relative_pupil_position((right_eye.x, right_eye.y), head_ref_points)

				# Компенсируем движение головы
				compensated_left = (
					relative_left[0] - head_movement[0],
					relative_left[1] - head_movement[1]
				)
				compensated_right = (
					relative_right[0] - head_movement[0],
					relative_right[1] - head_movement[1]
				)

				pupil_coords['left'] = left_eye_coords
				pupil_coords['right'] = right_eye_coords
				relative_pupil_coords['left'] = compensated_left
				relative_pupil_coords['right'] = compensated_right

				# Отрисовка зрачков
				cv2.circle(frame_with_pupils, left_eye_coords, 5, (0, 255, 0), -1)
				cv2.circle(frame_with_pupils, right_eye_coords, 5, (0, 0, 255), -1)

				# Обновление траектории взгляда (используем компенсированные координаты)
				self._update_gaze_trajectory(compensated_left, compensated_right)

		combined_frame = np.hstack((frame_with_pupils, self.trajectory_canvas))
		return combined_frame, pupil_coords, relative_pupil_coords

	def _update_gaze_trajectory(self, left_eye, right_eye):
		"""Обновление траектории взгляда с компенсированными координатами"""
		# Средняя точка между глазами (направление взгляда)
		gaze_x = (left_eye[0] + right_eye[0]) / 2
		gaze_y = (left_eye[1] + right_eye[1]) / 2

		# Преобразование координат взгляда в экранные
		screen_x = int((gaze_x + 0.5) * self.reduced_width)  # +0.5 для центра
		screen_y = int((gaze_y + 0.5) * self.reduced_height)

		# Сохранение траектории взгляда
		self.trajectory_screen.append((screen_x, screen_y))

		# Ограничение длины траектории
		if len(self.trajectory_screen) > 100:
			self.trajectory_screen.pop(0)

		# Рисование траектории
		self.trajectory_canvas.fill(0)  # Очищаем canvas
		for i in range(1, len(self.trajectory_screen)):
			cv2.line(self.trajectory_canvas, self.trajectory_screen[i - 1],
			         self.trajectory_screen[i], (255, 255, 255), 2)

		# Рисование текущей точки взгляда
		cv2.circle(self.trajectory_canvas, (screen_x, screen_y), 2, (0, 255, 0), -1)

