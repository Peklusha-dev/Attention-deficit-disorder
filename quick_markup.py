import os
import cv2
import pandas as pd
import numpy as np  # Добавляем импорт numpy

# Папка с изображениями
image_folder = './test_data/video/video2_frames'  # <-- замените на свою папку
output_csv = 'annotations_2.csv'

# Загружаем все изображения
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Параметры окна
screen_width = 1080  # Ваш экран по ширине
screen_height = 600  # Ваш экран по высоте

# Список для сохранения разметки
annotations = []

# Состояние
current_idx = 0
clicks = []
confirmed = False
zoom_factor = 1.0
offset_x, offset_y = 0, 0  # Для перемещения изображения
is_dragging = False
start_x, start_y = 0, 0

def mouse_callback(event, x, y, flags, param):
    global clicks, is_dragging, start_x, start_y, offset_x, offset_y, zoom_factor  # Указываем, что zoom_factor глобален
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicks) < 2:
            clicks.append(param['inverse_resize'](x, y))
        if len(clicks) == 0:
            is_dragging = True
            start_x, start_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_dragging:
            dx = x - start_x
            dy = y - start_y
            offset_x += dx
            offset_y += dy
            start_x, start_y = x, y
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags & cv2.EVENT_FLAG_CTRLKEY:  # Если зажата клавиша Ctrl
            if flags > 0:  # колесо вверх
                zoom_factor *= 1.1
            else:  # колесо вниз
                zoom_factor /= 1.1

def resize_to_screen(img):
    global zoom_factor
    h, w = img.shape[:2]
    scale = min(screen_width / w, screen_height / h, zoom_factor)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    def inverse_resize(x, y):
        return (int(x / scale), int(y / scale))
    return resized, inverse_resize

def save_annotations():
    if annotations:
        df = pd.DataFrame(annotations)
        df.to_csv(output_csv, index=False)
        print(f"Разметка сохранена в {output_csv}")
    else:
        print("Нет сохранённых данных.")

while True:
    img_path = os.path.join(image_folder, image_files[current_idx])
    img = cv2.imread(img_path)
    if img is None:
        print(f"Ошибка загрузки {img_path}")
        break

    display_img, inverse_resize = resize_to_screen(img.copy())
    display_img = cv2.warpAffine(display_img, np.float32([[1, 0, offset_x], [0, 1, offset_y]]), (display_img.shape[1], display_img.shape[0]))

    draw_img = display_img.copy()

    for idx, (x, y) in enumerate(clicks):
        cv2.circle(draw_img, (int(x * display_img.shape[1] / img.shape[1]), int(y * display_img.shape[0] / img.shape[0])), 5, (0, 255, 0), -1)
        cv2.putText(draw_img, str(idx+1), (int(x * display_img.shape[1] / img.shape[1])+5, int(y * display_img.shape[0] / img.shape[0])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Подписи для глаз
    cv2.putText(draw_img, "Left Eye", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(draw_img, "Right Eye", (display_img.shape[1] - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    instructions = "Click 2 points -> ENTER to confirm | BACKSPACE remove last | ESC exit | + Zoom in | - Zoom out | P Skip"
    cv2.putText(draw_img, instructions, (10, display_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Отображаем изображение
    cv2.imshow('Annotator', draw_img)
    cv2.setMouseCallback('Annotator', mouse_callback, param={'inverse_resize': inverse_resize})

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key == 8:  # Backspace
        if clicks:
            clicks.pop()
    elif key == 13:  # Enter
        if len(clicks) == 2:
            # Проверка, размечена ли уже картинка
            if any(annotation['filename'] == image_files[current_idx] for annotation in annotations):
                print("Предупреждение: эта картинка уже размечена!")
            else:
                annotations.append({
                    'filename': image_files[current_idx],
                    'left_x': clicks[0][0],
                    'left_y': clicks[0][1],
                    'right_x': clicks[1][0],
                    'right_y': clicks[1][1],
                })
                current_idx += 1
                clicks = []
                if current_idx >= len(image_files):
                    print("Разметка завершена.")
                    break
        else:
            print("Пожалуйста, поставьте 2 точки!")
    elif key == ord('+'):  # Zoom in
        zoom_factor *= 1.1
    elif key == ord('-'):  # Zoom out
        zoom_factor /= 1.1
    elif key == ord('d') or key == 83:  # стрелка вправо
        current_idx = min(current_idx + 1, len(image_files) - 1)
        clicks = []
    elif key == ord('a') or key == 81:  # стрелка влево
        current_idx = max(current_idx - 1, 0)
        clicks = []
    elif key == ord('p'):  # Skip (пропустить)
        print(f"Пропускаем изображение {image_files[current_idx]}")
        current_idx += 1
        clicks = []
        if current_idx >= len(image_files):
            print("Разметка завершена.")
            break

cv2.destroyAllWindows()

# Сохраняем результат
save_annotations()

