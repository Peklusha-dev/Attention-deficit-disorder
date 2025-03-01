import cv2
import mediapipe as mp
import numpy as np

# Инициализация Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Траектории для глаз
trajectory_screen = []

# Коэффициент масштабирования траектории
scaling_factor = 7

# Открытие видео
cap = cv2.VideoCapture('test_data/video4.mp4')  #0 - если хотите использовать изображение с веб-камеры

# Получение FPS видео
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)  # Задержка в миллисекундах

# Размеры окна
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Уменьшенный размер для отображения видео
reduced_width = frame_width // 2
reduced_height = frame_height // 2

# Полотно для траектории взгляда
trajectory_canvas = np.zeros((reduced_height, reduced_width, 3), dtype=np.uint8)


# Функция масштабирования и центрирования координат
def scale_and_center_coords(x, y, scaling_factor, canvas_width, canvas_height):
	"""
    Масштабирует и центрирует координаты (x, y) с учетом коэффициента масштабирования
    и размеров полотна (canvas_width, canvas_height).
    """
	center_x, center_y = canvas_width // 2, canvas_height // 2
	scaled_x = int((x - center_x) * scaling_factor + center_x)
	scaled_y = int((y - center_y) * scaling_factor + center_y)

	# Ограничение, чтобы координаты не выходили за пределы окна
	scaled_x = min(max(scaled_x, 0), canvas_width - 1)
	scaled_y = min(max(scaled_y, 0), canvas_height - 1)

	return scaled_x, scaled_y


# Основной цикл обработки видео
while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break

	# Преобразование кадра для Mediapipe
	rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	result = face_mesh.process(rgb_frame)

	# Уменьшение размера кадра
	frame_with_pupils = cv2.resize(frame, (reduced_width, reduced_height))

	if result.multi_face_landmarks:
		for face_landmarks in result.multi_face_landmarks:
			# Индексы для зрачков в Mediapipe
			left_eye_idx = 468  # Левый зрачок
			right_eye_idx = 473  # Правый зрачок

			# Получение координат зрачков
			left_eye = face_landmarks.landmark[left_eye_idx]
			right_eye = face_landmarks.landmark[right_eye_idx]

			# Преобразование координат для отрисовки зрачков
			left_eye_coords = (int(left_eye.x * reduced_width), int(left_eye.y * reduced_height))
			right_eye_coords = (int(right_eye.x * reduced_width), int(right_eye.y * reduced_height))

			# Отрисовка зрачков на видео
			cv2.circle(frame_with_pupils, left_eye_coords, 3, (0, 255, 0), -1)
			cv2.circle(frame_with_pupils, right_eye_coords, 3, (0, 0, 255), -1)

			# Средняя точка между глазами (нормализованные координаты)
			gaze_x = (left_eye.x + right_eye.x) / 2
			gaze_y = (left_eye.y + right_eye.y) / 2

			# Преобразование нормализованных координат в пиксели (уменьшенный размер)
			display_x = int(gaze_x * reduced_width)
			display_y = int(gaze_y * reduced_height)

			# Масштабирование и центрирование координат
			scaled_x, scaled_y = scale_and_center_coords(display_x, display_y, scaling_factor, reduced_width,
			                                             reduced_height)

			# Сохранение траектории взгляда
			trajectory_screen.append((scaled_x, scaled_y))

			# Ограничение длины траектории
			if len(trajectory_screen) > 100:
				trajectory_screen.pop(0)

			# Отрисовка траектории
			for i in range(1, len(trajectory_screen)):
				cv2.line(trajectory_canvas, trajectory_screen[i - 1], trajectory_screen[i], (255, 255, 255))

			# Рисование текущей точки взгляда
			cv2.circle(trajectory_canvas, (scaled_x, scaled_y), 3, (0, 255, 0))

	# Объединение видео и траектории в одно окно
	combined_frame = np.hstack((frame_with_pupils, trajectory_canvas))

	# Отображение результатов
	cv2.imshow("Video and Eye Trajectory", combined_frame)

	if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
