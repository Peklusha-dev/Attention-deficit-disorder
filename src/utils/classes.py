import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# 1. Загружаем файл
mat_data = scipy.io.loadmat(r'C:\C++_projects\EyeMovementDetectorEvaluation\annotated_data\data used in the article\img\TH34_img_Europe_labelled_MN.mat')

# 2. Извлекаем сырые данные и результаты
et_data = mat_data['ETdata']  # Основные данные (N, 6)
results = mat_data['results']  # Структура с метками

# 3. Исследуем структуру 'results'
print("Тип results:", type(results))
print("Форма results:", results.shape)
print("\nПоля структуры results[0, 0]:")
results_struct = results[0, 0]

for field_name in results_struct.dtype.names:
	field_data = results_struct[field_name][0]  # Убрали один [0, 0], т.к. массив 1D
	print(f"\n--- Поле: {field_name} ---")
	print(f"  Тип: {type(field_data)}")
	print(f"  Форма: {field_data.shape if hasattr(field_data, 'shape') else 'N/A'}")

	# Пробуем показать содержимое
	if isinstance(field_data, np.ndarray):
		if field_data.dtype.names:  # Если это вложенная структура
			print(f"  Это вложенная структура с полями: {field_data.dtype.names}")
			# Исследуем первый элемент вложенной структуры
			nested = field_data[0]
			if hasattr(nested, 'dtype') and nested.dtype.names:
				print(f"  Поля первого элемента: {nested.dtype.names}")
		elif field_data.dtype == object:  # Массив объектов (ячеек)
			print(f"  Содержимое первого элемента: {field_data[0]}")
			# Покажем тип и форму первых нескольких элементов
			for i in range(min(3, len(field_data))):
				elem = field_data[i]
				if isinstance(elem, np.ndarray):
					print(f"    Элемент {i}: тип={type(elem)}, форма={elem.shape}")
				else:
					print(f"    Элемент {i}: тип={type(elem)}, значение={elem}")
		else:  # Обычный числовой массив
			print(f"  Первые 5 элементов: {field_data[:5] if len(field_data) > 5 else field_data}")
	else:
		print(f"  Значение: {field_data}")

# 4. Особое внимание полю 'tagTime' - скорее всего, там метки событий
print("\n" + "=" * 50)
print("ДЕТАЛЬНОЕ ИССЛЕДОВАНИЕ tagTime:")
tagTime_data = results_struct['tagTime'][0]

if isinstance(tagTime_data, np.ndarray) and len(tagTime_data) > 0:
	print(f"Количество элементов в tagTime: {len(tagTime_data)}")

	# Покажем структуру первого элемента tagTime
	first_tag = tagTime_data[0]
	print(f"\nПервый элемент tagTime:")
	print(f"  Тип: {type(first_tag)}")

	if isinstance(first_tag, np.ndarray):
		print(f"  Форма: {first_tag.shape}")
		print(f"  dtype: {first_tag.dtype}")

		# Если это структура
		if first_tag.dtype.names:
			print(f"  Поля структуры: {first_tag.dtype.names}")
			# Покажем значения полей для первого события
			for name in first_tag.dtype.names:
				value = first_tag[name][0, 0]  # Теперь используем [0, 0] для элемента структуры
				print(f"    {name}: {value}")

		# Если это обычный массив
		elif first_tag.dtype == object:
			print(f"  Содержимое: {first_tag}")
			if len(first_tag) > 0:
				print(f"  Элемент 0: {first_tag[0]}")

	# Если это скаляр или другой тип
	else:
		print(f"  Значение: {first_tag}")

	# Покажем еще несколько элементов для понимания паттерна
	print(f"\nСледующие элементы tagTime (первые {min(5, len(tagTime_data))}):")
	for i in range(min(5, len(tagTime_data))):
		elem = tagTime_data[i]
		if isinstance(elem, np.ndarray) and elem.dtype.names:
			event_info = {name: elem[name][0, 0] for name in elem.dtype.names}
			print(f"  Элемент {i}: {event_info}")
		else:
			print(f"  Элемент {i}: {elem}")
else:
	print("tagTime пуст или не является массивом")