from psychopy import visual, core, monitors
import pandas as pd

# Настройка монитора
monitor = monitors.Monitor(name='zephyrus_g14')
monitor.setSizePix((2560, 1440))  # Замените на (2560, 1600) для WQXGA
monitor.setWidth(30.95)  # см
monitor.setDistance(38)  # см

# Создание окна
win = visual.Window(
	size=(1280, 720),
	monitor=monitor,
	units='deg',
	fullscr=True
)

# Калибровочные точки (3x3 сетка, ±4° по x и y)
calibration_points = [
	(-4, 4), (0, 4), (4, 4),
	(-4, 0), (0, 0), (4, 0),
	(-4, -4), (0, -4), (4, -4)
]

# Стимул (точка фиксации)
fixation = visual.Circle(
	win,
	radius=0.5,  # Радиус 0.5°
	fillColor='white'
)

# Сохранение данных
data = []
start_time = core.getTime()

# Мигающий экран для синхронизации
for _ in range(3):
	win.color = 'white'
	win.flip()
	core.wait(0.2)
	win.color = 'black'
	win.flip()
	core.wait(0.2)

# Показ точек
for point in calibration_points:
	fixation.pos = point
	point_start = core.getTime()

	# Показ точки 3 секунды
	while core.getTime() - point_start < 3.0:
		fixation.draw()
		win.flip()

	# Сохранение данных
	data.append({
		'point_x_deg': point[0],
		'point_y_deg': point[1],
		'time': point_start - start_time
	})

# Закрытие
win.close()

# Сохранение в CSV
df = pd.DataFrame(data)
df.to_csv('calibration_points.csv', index=False)
print("Калибровочные точки сохранены в calibration_points.csv")

core.quit()