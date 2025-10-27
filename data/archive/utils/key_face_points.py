import cv2
import dlib

# Укажите путь к видеофайлу
video_path = '../test_data/video4.mp4'  # Замените на путь к вашему видеофайлу

# Открытие видеофайла
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка при открытии видеофайла")
    exit()

# Подключение детектора лиц и предсказателя ключевых точек
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")

while True:
    # Получаем кадр из видео
    ret, frame = cap.read()
    if not ret:
        break  # Если видео закончилось, выходим из цикла

    # Конвертируем кадр в черно-белое изображение для детектора лиц
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц
    faces = detector(grayFrame)

    # Обработка каждого найденного лица
    for face in faces:
        # Выводим количество лиц на изображении
        cv2.putText(frame, "{} face(s) found".format(len(faces)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Получаем координаты прямоугольника, обрамляющего лицо, и рисуем его
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Получаем координаты ключевых точек на лице и рисуем их
        landmarks = predictor(grayFrame, face)
        for n in range(0, 68):  # 68 ключевых точек лица
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

    # Выводим кадр с наложенными ключевыми точками
    cv2.putText(frame, "Press ESC to close frame", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow("Frame", frame)

    # Выход из цикла при нажатии ESC
    key = cv2.waitKey(1)
    if key == 27:
        break

# Закрытие видеофайла и окон
cap.release()
cv2.destroyAllWindows()
