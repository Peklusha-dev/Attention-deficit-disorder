import cv2
import mediapipe as mp
import math

# Инициализация Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Открытие видео
cap = cv2.VideoCapture('../test_data/video4.mp4')  # Замените на 0 для веб-камеры
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Функция для вычисления угла наклона головы
def get_head_rotation(landmarks, w, h):
    # Используем ключевые точки для вычисления наклона головы (например, нос, глаза)
    nose = landmarks.landmark[1]  # Координаты носа
    left_eye = landmarks.landmark[33]  # Координаты левого глаза
    right_eye = landmarks.landmark[263]  # Координаты правого глаза

    # Преобразуем координаты в пиксели
    nose_x, nose_y = int(nose.x * w), int(nose.y * h)
    left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
    right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)

    # Расчет угла наклона головы между глазом и носом
    dx = right_eye_x - left_eye_x
    dy = right_eye_y - left_eye_y
    angle = math.degrees(math.atan2(dy, dx))

    return angle, (nose_x, nose_y), (left_eye_x, left_eye_y), (right_eye_x, right_eye_y)

# Обработка кадров
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Обработка кадра в Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)
    h, w, _ = frame.shape

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Получаем угол наклона головы и координаты точек
            angle, nose, left_eye, right_eye = get_head_rotation(face_landmarks, w, h)

            # Рисуем линии, указывающие наклон головы
            cv2.line(frame, nose, left_eye, (0, 255, 0), 2)  # Линия от носа к левому глазу
            cv2.line(frame, nose, right_eye, (0, 255, 0), 2)  # Линия от носа к правому глазу

            # Добавляем текст с углом наклона
            cv2.putText(frame, f"Head Tilt: {angle:.2f} degrees", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Отображаем кадр
    cv2.imshow('Head Tilt Visualization', frame)

    # Ожидание нажатия клавиши для продолжения
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрытие видео
cap.release()
cv2.destroyAllWindows()
