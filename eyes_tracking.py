import cv2
import mediapipe as mp
import numpy as np

# Инициализация Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Траектории для глаз
trajectory_screen = []

# Открытие видео
cap = cv2.VideoCapture('test_data/video4.mp4')  # Замените 'video4.mp4' на 0 для веб-камеры

# Получение FPS видео
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)  # Задержка в миллисекундах

# Размеры окна
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Уменьшенный размер для отображения видео
reduced_width = frame_width // 2
reduced_height = frame_height // 2

# Размер экрана (1920x1080 пикселей)
screen_width, screen_height = 1920, 1080

# Полотно для траектории взгляда
trajectory_canvas = np.zeros((reduced_height, reduced_width, 3), dtype=np.uint8)

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
            h, w, _ = frame.shape
            left_eye = face_landmarks.landmark[left_eye_idx]
            right_eye = face_landmarks.landmark[right_eye_idx]

            left_eye_coords = (int(left_eye.x * reduced_width), int(left_eye.y * reduced_height))
            right_eye_coords = (int(right_eye.x * reduced_width), int(right_eye.y * reduced_height))

            # Отрисовка зрачков
            cv2.circle(frame_with_pupils, left_eye_coords, 5, (0, 255, 0), -1)
            cv2.circle(frame_with_pupils, right_eye_coords, 5, (0, 0, 255), -1)

            # Средняя точка между глазами (направление взгляда)
            gaze_x = (left_eye.x + right_eye.x) / 2
            gaze_y = (left_eye.y + right_eye.y) / 2

            # Преобразование координат взгляда в экранные
            screen_x = int(gaze_x * reduced_width)
            screen_y = int(gaze_y * reduced_height)

            # Сохранение траектории взгляда
            trajectory_screen.append((screen_x, screen_y))

            # Ограничение длины траектории
            if len(trajectory_screen) > 100:
                trajectory_screen.pop(0)

            # Рисование траектории на экране
            for i in range(1, len(trajectory_screen)):
                cv2.line(trajectory_canvas, trajectory_screen[i - 1], trajectory_screen[i], (255, 255, 255), 2)

            # Рисование текущей точки взгляда
            cv2.circle(trajectory_canvas, (screen_x, screen_y), 2, (0, 255, 0), -1)

    # Объединение видео и траектории в одно окно
    combined_frame = np.hstack((frame_with_pupils, trajectory_canvas))

    # Отображение результатов
    cv2.imshow("Video and Eye Trajectory", combined_frame)

    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
