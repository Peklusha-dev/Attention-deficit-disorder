import cv2
import numpy as np
from pupil_head_detector import PupilHeadDetector


def test_webcam():
	"""Тестирование с веб-камерой в реальном времени"""
	detector = PupilHeadDetector()
	cap = cv2.VideoCapture(0)  # Используем веб-камеру

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		# Обработка кадра
		result_frame, abs_coords, rel_coords = detector.detect_pupils_in_frame(frame)

		# Вывод координат
		print(f"Абсолютные: {abs_coords} | Относительные: {rel_coords}")

		# Отображение результата
		cv2.imshow('Eye Tracking', result_frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


def test_video_file(video_path):
	"""Тестирование на видеофайле"""
	detector = PupilHeadDetector()
	cap = cv2.VideoCapture(video_path)

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		result_frame, abs_coords, rel_coords = detector.detect_pupils_in_frame(frame)
		cv2.imshow('Video Test', result_frame)

		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	print("Выберите режим тестирования:")
	print("1 - Веб-камера (режим реального времени)")
	print("2 - Видеофайл")

	choice = input("Ваш выбор (1/2): ").strip()

	if choice == '1':
		test_webcam()
	elif choice == '2':
		test_video_file('test_data/video2.mp4')
	else:
		print("Некорректный выбор. Завершение работы.")