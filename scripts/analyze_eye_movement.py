from eye_tracker.detector import PupilDetector
from eye_tracker.calibration import Calibrator
from eye_tracker.data_processing import smooth_coordinates, compute_velocity, detect_saccades_fixations
from eye_tracker.visualization import plot_eye_movement, plot_heatmap
from eye_tracker.utils import load_pupil_data
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def analyze_eye_movement(input_data, output_csv="pupil_data.csv", output_plot="eye_movement.html",
                         output_heatmap="heatmap.png", L_mm=177, D_mm=380, head_tracking=True,
                         velocity_threshold=100, min_fixation_duration=0.1):
	"""
	Анализирует движение глаз, строит графики позиций, скоростей и саккад/фиксаций.
	:param input_data: Путь к видео или папке с изображениями
	:param output_csv: Путь к CSV-файлу для сохранения координат
	:param output_plot: Путь к HTML-файлу для графика
	:param output_heatmap: Путь к PNG-файлу для тепловой карты
	:param L_mm, D_mm: Параметры калибровки (расстояния в мм)
	:param head_tracking: Включить компенсацию движения головы
	:param velocity_threshold: Порог скорости для детекции саккад (градусы/с)
	:param min_fixation_duration: Минимальная длительность фиксации (с)
	"""
	# 1. Обработка входных данных
	detector = PupilDetector(head_tracking=head_tracking)
	input_path = Path(input_data)

	if input_path.is_file() and input_path.suffix.lower() in ['.mov', '.mp4', '.avi']:
		detector.process_video(input_data, output_csv=output_csv, show_result=False)
	elif input_path.is_dir():
		detector.process_image_folder(input_data, output_csv=output_csv, output_image_folder=None)
	else:
		raise ValueError("Input must be a video file or a folder with images")

	# 2. Загрузка данных
	df = load_pupil_data(output_csv)
	if df.empty:
		print(f"No data found in {output_csv}")
		return

	# Используем относительные координаты, если включена компенсация головы
	x_col = "relative_left_x" if head_tracking and "relative_left_x" in df else "left_x"
	y_col = "relative_left_y" if head_tracking and "relative_left_y" in df else "left_y"
	avg_x = (df[x_col] + df["right_x"]) / 2
	avg_y = (df[y_col] + df["right_y"]) / 2
	time = df["time"]

	# 3. Сглаживание координат
	x_smooth, y_smooth = smooth_coordinates(avg_x, avg_y, window_length=11, polyorder=2)

	# 4. Калибровка
	calibrator = Calibrator()
	calibrator.calculate_calibration(L_mm, D_mm, x1=avg_x.iloc[0], y1=avg_y.iloc[0],
	                                 x2=avg_x.iloc[-1], y2=avg_y.iloc[-1])
	theta_x, theta_y = calibrator.calibrate_coordinates(x_smooth, y_smooth, avg_x.iloc[0], avg_y.iloc[0])

	# 5. Вычисление скорости
	velocity = compute_velocity(theta_x, theta_y, time)

	# 6. Детекция саккад и фиксаций
	events = detect_saccades_fixations(time, theta_x, theta_y, velocity_threshold, min_fixation_duration)

	# 7. Построение графиков
	fig = make_subplots(
		rows=4, cols=1,
		subplot_titles=("Horizontal Position (px)", "Horizontal Velocity (deg/s)",
		                "Vertical Position (px)", "Vertical Velocity (deg/s)"),
		shared_xaxes=True, vertical_spacing=0.1
	)

	# Горизонтальная позиция
	fig.add_trace(go.Scatter(x=time, y=x_smooth, mode='lines+markers', name='Horizontal (px)',
	                         line=dict(color='blue')), row=1, col=1)
	fig.add_trace(go.Scatter(x=time, y=theta_x, mode='lines', name='Horizontal Angle (deg)',
	                         line=dict(color='green', dash='dash')), row=1, col=1)

	# Горизонтальная скорость
	fig.add_trace(go.Scatter(x=time, y=velocity, mode='lines', name='Horizontal Velocity (deg/s)',
	                         line=dict(color='cyan')), row=2, col=1)

	# Вертикальная позиция
	fig.add_trace(go.Scatter(x=time, y=y_smooth, mode='lines+markers', name='Vertical (px)',
	                         line=dict(color='orange')), row=3, col=1)
	fig.add_trace(go.Scatter(x=time, y=theta_y, mode='lines', name='Vertical Angle (deg)',
	                         line=dict(color='red', dash='dash')), row=3, col=1)

	# Вертикальная скорость
	fig.add_trace(go.Scatter(x=time, y=velocity, mode='lines', name='Vertical Velocity (deg/s)',
	                         line=dict(color='magenta')), row=4, col=1)

	# Добавление саккад и фиксаций
	for _, event in events.iterrows():
		color = 'red' if event['type'] == 'saccade' else 'green'
		alpha = 0.3
		for row in [1, 3]:  # Добавляем области на графики позиций
			fig.add_vrect(x0=event['start_time'], x1=event['end_time'], fillcolor=color, opacity=alpha,
			              line_width=0, row=row, col=1)

	fig.update_xaxes(title_text="Time (s)", row=4, col=1)
	fig.update_yaxes(title_text="Position (pixels)", row=1, col=1)
	fig.update_yaxes(title_text="Velocity (deg/s)", row=2, col=1)
	fig.update_yaxes(title_text="Position (pixels)", row=3, col=1)
	fig.update_yaxes(title_text="Velocity (deg/s)", row=4, col=1)
	fig.update_layout(title="Eye Movement Analysis with Saccades and Fixations",
	                  showlegend=True, height=800, width=800, hovermode="closest")
	fig.write_html(output_plot)
	print(f"График сохранен в {output_plot}")

	# 8. Построение тепловой карты фиксаций
	fixations = events[events['type'] == 'fixation']
	if not fixations.empty:
		plot_heatmap(fixations['mean_x'], fixations['mean_y'], output_file=output_heatmap)
		print(f"Тепловая карта сохранена в {output_heatmap}")


if __name__ == "__main__":
	# Пример для видео
	'''analyze_eye_movement('test_data/video_7.MOV',
	                     output_csv='output/pupil_data/pupil_data_7.csv',
	                     output_plot='output/plots/eye_movement_7.html',
	                     output_heatmap='output/plots/heatmap_7.png')'''

	# Пример для изображений
	analyze_eye_movement('../test_data/video/video7_frames',
	                     output_csv='../output/pupil_data/pupil_data_images7.csv',
	                     output_plot='../output/plots/eye_movement_images7.html',
	                     output_heatmap='../output/plots/heatmap_images7.png')