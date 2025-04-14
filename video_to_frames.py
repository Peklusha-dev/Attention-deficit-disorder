import cv2
import os
from datetime import timedelta


def video_to_frames(video_path, output_folder, fps=None, time_format="%H_%M_%S_%f"):
	"""
	Нарезает видео на кадры и сохраняет с именами по времени

	:param video_path: путь к видеофайлу
	:param output_folder: папка для сохранения кадров
	:param fps: частота кадров (если None - исходная частота)
	:param time_format: формат времени в имени файла
	"""
	# Создаем папку для результатов
	os.makedirs(output_folder, exist_ok=True)

	# Открываем видео
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise ValueError(f"Не удалось открыть видео: {video_path}")

	# Получаем параметры видео
	original_fps = cap.get(cv2.CAP_PROP_FPS)
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	duration = frame_count / original_fps

	# Используем указанный FPS или оригинальный
	target_fps = fps if fps is not None else original_fps
	frame_interval = int(round(original_fps / target_fps))

	print(f"Обработка видео: {video_path}")
	print(f"Длительность: {timedelta(seconds=duration)}")
	print(f"Частота кадров: {original_fps} -> {target_fps} (сохраняем каждый {frame_interval}-й кадр)")

	frame_number = 0
	saved_count = 0

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		# Пропускаем кадры согласно заданной частоте
		if frame_number % frame_interval == 0:
			# Вычисляем время текущего кадра
			milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
			seconds = milliseconds / 1000
			time_obj = timedelta(seconds=seconds)

			# Форматируем время для имени файла
			time_str = str(time_obj).replace(":", "_").replace(".", "_")
			filename = f"frame_{time_str}.jpg"
			output_path = os.path.join(output_folder, filename)

			# Сохраняем кадр
			cv2.imwrite(output_path, frame)
			saved_count += 1

		frame_number += 1

		# Прогресс
		if frame_number % 100 == 0:
			print(f"Обработано: {frame_number}/{frame_count} кадров")

	cap.release()
	print(f"Готово! Сохранено {saved_count} кадров в {output_folder}")


if __name__ == "__main__":
	# Пример 1: Сохранить все кадры с оригинальной частотой
	video_to_frames("test_data/video1.mp4", "test_data/video1_frames")
	# Пример 2: Сохранять 1 кадр в секунду
	#video_to_frames("input.mp4", "output_frames_1fps", fps=1)