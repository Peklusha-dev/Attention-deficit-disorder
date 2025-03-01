import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt

# Инициализация Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Открытие видео
cap = cv2.VideoCapture('test_data/video4.mp4')  # Замените на 0 для веб-камеры
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Хранилища данных для графика
time_data = []
left_eye_angles = []
right_eye_angles = []
frame_counter = 0

# Расстояние до камеры (в миллиметрах)
distance_to_camera_mm = 500  # 50 см = 500 мм

# Функция для вычисления угла смещения
def calculate_angle_with_distance(inner, outer, pupil, distance):
    # Центр глаза
    eye_center_x = (inner[0] + outer[0]) / 2
    eye_center_y = (inner[1] + outer[1]) / 2
    # Смещения зрачка
    delta_x = pupil[0] - eye_center_x
    delta_y = pupil[1] - eye_center_y
    # Угол смещения
    angle = math.degrees(math.atan(math.sqrt(delta_x**2 + delta_y**2) / distance))
    return angle

# Обработка кадров
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Обработка кадра в Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)
    h, w, _ = frame.shape

    left_angle, right_angle = 0, 0  # Значения по умолчанию
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Координаты для левого глаза
            left_eye_outer = face_landmarks.landmark[33]
            left_eye_inner = face_landmarks.landmark[133]
            left_pupil = face_landmarks.landmark[468]

            # Координаты для правого глаза
            right_eye_outer = face_landmarks.landmark[362]
            right_eye_inner = face_landmarks.landmark[263]
            right_pupil = face_landmarks.landmark[473]

            # Конвертация координат в пиксели
            def to_pixel_coords(landmark):
                return (int(landmark.x * w), int(landmark.y * h))

            left_eye_outer_px = to_pixel_coords(left_eye_outer)
            left_eye_inner_px = to_pixel_coords(left_eye_inner)
            left_pupil_px = to_pixel_coords(left_pupil)

            right_eye_outer_px = to_pixel_coords(right_eye_outer)
            right_eye_inner_px = to_pixel_coords(right_eye_inner)
            right_pupil_px = to_pixel_coords(right_pupil)

            # Расчет угла смещения для каждого глаза
            left_angle = calculate_angle_with_distance(left_eye_inner_px, left_eye_outer_px, left_pupil_px, distance_to_camera_mm)
            right_angle = calculate_angle_with_distance(right_eye_inner_px, right_eye_outer_px, right_pupil_px, distance_to_camera_mm)

    # Добавление данных
    current_time = frame_counter / fps
    time_data.append(current_time)
    left_eye_angles.append(left_angle)
    right_eye_angles.append(right_angle)
    frame_counter += 1

cap.release()

# Построение и сохранение графика
plt.figure(figsize=(12, 6))
plt.plot(time_data, left_eye_angles, label='Left Eye Angle', color='green')
plt.plot(time_data, right_eye_angles, label='Right Eye Angle', color='blue')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Eye Angle (degrees)')
plt.legend()
plt.title('Eye Angle Over Time')
plt.grid(alpha=0.3)
plt.savefig('Angles_video4.png', dpi=300)
