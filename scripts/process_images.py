from eye_tracker.detector import PupilDetector
from eye_tracker.calibration import Calibrator
from eye_tracker.data_processing import smooth_coordinates, compute_velocity, detect_saccades_fixations
from eye_tracker.visualization import plot_eye_movement, plot_heatmap
from eye_tracker.utils import load_pupil_data, save_pupil_data
from pathlib import Path


def process_images(input_folder, output_csv="pupil_data.csv", output_image_folder=None, L_mm=177, D_mm=380, head_tracking=True):
    """
    Обработка изображений в папке, калибровка, анализ саккад/фиксаций и визуализация.
    :param input_folder: Путь к папке с изображениями
    :param output_csv: Путь к CSV-файлу для сохранения координат
    :param output_image_folder: Путь к папке для сохранения обработанных изображений
    :param L_mm, D_mm: Параметры калибровки (расстояния в мм)
    :param head_tracking: Включить компенсацию движения головы
    """
    # Детекция зрачков
    detector = PupilDetector(head_tracking=head_tracking)
    pupil_data = detector.process_image_folder(input_folder, output_csv, output_image_folder)

    # Загрузка данных
    df = load_pupil_data(output_csv)
    avg_x = (df["left_x"] + df["right_x"]) / 2
    avg_y = (df["left_y"] + df["right_y"]) / 2
    time = df["time"]

    # Сглаживание
    x_smooth, y_smooth = smooth_coordinates(avg_x, avg_y)

    # Калибровка
    calibrator = Calibrator()
    calibrator.calculate_calibration(L_mm, D_mm, x1=avg_x.iloc[0], y1=avg_y.iloc[0], x2=avg_x.iloc[-1], y2=avg_y.iloc[-1])
    theta_x, theta_y = calibrator.calibrate_coordinates(x_smooth, y_smooth, avg_x.iloc[0], avg_y.iloc[0])

    # Вычисление скорости
    velocity = compute_velocity(theta_x, theta_y, time)

    # Детекция саккад и фиксаций
    events = detect_saccades_fixations(time, theta_x, theta_y, velocity_threshold=100)

    # Визуализация
    output_plot = str(Path(output_csv).parent / "plots" / f"{Path(output_csv).stem}_movement.html")
    plot_eye_movement(time, x_smooth, y_smooth, theta_x, theta_y, velocity, velocity, output_file=output_plot)

    fixations = events[events['type'] == 'fixation']
    heatmap_path = str(Path(output_csv).parent / "plots" / f"{Path(output_csv).stem}_heatmap.png")
    plot_heatmap(fixations['mean_x'], fixations['mean_y'], output_file=heatmap_path)

    print(f"Обработка завершена. Данные сохранены в {output_csv}, графики в {output_plot}, тепловая карта в {heatmap_path}")

if __name__ == "__main__":
    process_images('test_data/images', output_csv='output/pupil_data/pupil_data_images.csv', output_image_folder='output/processed_images')