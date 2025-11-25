from eye_tracker.detector import PupilDetector
from eye_tracker.calibration import Calibrator
from eye_tracker.data_processing import smooth_coordinates, compute_velocity, detect_saccades_fixations
from eye_tracker.visualization import plot_eye_movement, plot_heatmap
from eye_tracker.utils import load_pupil_data


def analyze_data(csv_file, output_plot="eye_movement.html", L_mm=177, D_mm=380):
    df = load_pupil_data(csv_file)
    avg_x = (df["left_x"] + df["right_x"]) / 2
    avg_y = (df["left_y"] + df["right_y"]) / 2
    time = df["time"]

    # Сглаживание
    x_smooth, y_smooth = smooth_coordinates(avg_x, avg_y)

    # Калибровка
    calibrator = Calibrator()
    calibrator.calculate_calibration(L_mm, D_mm, x1=avg_x.iloc[0], y1=avg_y.iloc[0], x2=avg_x.iloc[-1], y2=avg_y.iloc[-1])
    theta_x, theta_y = calibrator.calibrate_coordinates(x_smooth, y_smooth, avg_x.iloc[0], avg_y.iloc[0])

    # Скорости
    velocity = compute_velocity(theta_x, theta_y, time)

    # Детекция саккад и фиксаций
    events = detect_saccades_fixations(time, theta_x, theta_y, velocity_threshold=100)

    # Визуализация
    plot_eye_movement(time, x_smooth, y_smooth, theta_x, theta_y, velocity, velocity, output_file=output_plot)
    fixations = events[events['type'] == 'fixation']
    plot_heatmap(fixations['mean_x'], fixations['mean_y'], output_file="output/plots/heatmap.png")


if __name__ == "__main__":
    analyze_data('output/pupil_data/pupil_data_7.csv', output_plot='output/plots/eye_movement_7.html')