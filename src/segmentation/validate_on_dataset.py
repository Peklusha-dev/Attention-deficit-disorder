import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os


@dataclass
class DetailedTestResult:
	"""Детальные результаты тестирования"""
	algorithm_name: str
	accuracy: float
	precision: float
	recall: float
	f1_score: float
	confusion_matrix: np.ndarray
	classification_report: str
	event_distribution: Dict
	file_results: Dict


class AdvancedEyeMovementTester:
	"""
	Улучшенный тестер с детальной диагностикой
	"""

	def __init__(self, data_folder: str):
		self.data_folder = data_folder
		self.datasets = {}

	def load_dataset(self, file_path: str) -> Dict[str, Any]:
		"""Загрузка датасета (как в предыдущей версии)"""
		data = scipy.io.loadmat(file_path)
		et_data = data['ETdata'][0, 0]
		raw_data = et_data[0]  # Основной массив 1658×6

		return {
			'x': raw_data[:, 3],
			'y': raw_data[:, 4],
			'labels': raw_data[:, 5],
			'timestamps': raw_data[:, 0],
			'sampling_rate': 500,
			'filename': os.path.basename(file_path)
		}

	def load_all_datasets(self, max_files: int = None) -> None:
		"""Загрузка всех датасетов"""
		categories = ['dots', 'img', 'video']
		self.datasets = {}

		for category in categories:
			category_path = os.path.join(self.data_folder, category)
			if os.path.exists(category_path):
				files = [f for f in os.listdir(category_path) if f.endswith('.mat')]
				if max_files:
					files = files[:max_files]

				self.datasets[category] = []
				for file in files:
					try:
						dataset = self.load_dataset(os.path.join(category_path, file))
						self.datasets[category].append(dataset)
					except Exception as e:
						print(f"Ошибка загрузки {file}: {e}")

	def analyze_algorithm_behavior(self, x: np.ndarray, y: np.ndarray,
	                               algorithm: Callable, **kwargs) -> Dict:
		"""Анализ поведения алгоритма на одном файле"""
		predictions = algorithm(x, y, **kwargs)

		# Анализ скорости для понимания порогов
		dx = np.diff(x, prepend=x[0])
		dy = np.diff(y, prepend=y[0])
		velocity = np.sqrt(dx ** 2 + dy ** 2)

		return {
			'predictions': predictions,
			'velocity_stats': {
				'mean': np.mean(velocity),
				'std': np.std(velocity),
				'min': np.min(velocity),
				'max': np.max(velocity),
				'percentile_50': np.percentile(velocity, 50),
				'percentile_90': np.percentile(velocity, 90),
				'percentile_95': np.percentile(velocity, 95)
			},
			'prediction_stats': {
				'fixation_ratio': np.mean(predictions),
				'total_fixations': np.sum(predictions),
				'total_non_fixations': len(predictions) - np.sum(predictions)
			}
		}

	def test_algorithm_detailed(self,
	                            algorithm: Callable,
	                            algorithm_name: str,
	                            target_events: List[int] = [1],
	                            **algorithm_kwargs) -> DetailedTestResult:
		"""
		Детальное тестирование алгоритма с диагностикой
		"""
		all_predictions = []
		all_ground_truth = []
		file_results = {}
		event_counts = {'true_positives': 0, 'false_positives': 0,
		                'true_negatives': 0, 'false_negatives': 0}

		for category, datasets in self.datasets.items():
			for dataset in datasets:
				ground_truth = (dataset['labels'] == target_events[0]).astype(int)

				# Анализ поведения алгоритма
				analysis = self.analyze_algorithm_behavior(
					dataset['x'], dataset['y'], algorithm, **algorithm_kwargs
				)

				predictions = analysis['predictions']

				# Собираем метрики для этого файла
				file_metrics = {}
				if len(predictions) == len(ground_truth):
					accuracy = np.mean(predictions == ground_truth)
					precision = self.safe_division(
						np.sum((predictions == 1) & (ground_truth == 1)),
						np.sum(predictions == 1)
					)
					recall = self.safe_division(
						np.sum((predictions == 1) & (ground_truth == 1)),
						np.sum(ground_truth == 1)
					)

					file_metrics = {
						'accuracy': accuracy,
						'precision': precision,
						'recall': recall,
						'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
						'velocity_stats': analysis['velocity_stats'],
						'prediction_stats': analysis['prediction_stats'],
						'true_fixation_ratio': np.mean(ground_truth)
					}

					all_predictions.extend(predictions)
					all_ground_truth.extend(ground_truth)

				file_results[dataset['filename']] = file_metrics

		# Общие метрики
		all_predictions = np.array(all_predictions)
		all_ground_truth = np.array(all_ground_truth)

		accuracy = np.mean(all_predictions == all_ground_truth)
		precision = self.safe_division(
			np.sum((all_predictions == 1) & (all_ground_truth == 1)),
			np.sum(all_predictions == 1)
		)
		recall = self.safe_division(
			np.sum((all_predictions == 1) & (all_ground_truth == 1)),
			np.sum(all_ground_truth == 1)
		)
		f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

		cm = confusion_matrix(all_ground_truth, all_predictions, labels=[0, 1])
		cr = classification_report(all_ground_truth, all_predictions,
		                           target_names=['Другие', 'Целевые'],
		                           zero_division=0)

		# Анализ распределения событий
		event_distribution = {
			'total_samples': len(all_ground_truth),
			'true_fixations': np.sum(all_ground_truth == 1),
			'true_non_fixations': np.sum(all_ground_truth == 0),
			'predicted_fixations': np.sum(all_predictions == 1),
			'predicted_non_fixations': np.sum(all_predictions == 0),
			'fixation_ratio_true': np.mean(all_ground_truth == 1),
			'fixation_ratio_pred': np.mean(all_predictions == 1)
		}

		return DetailedTestResult(
			algorithm_name=algorithm_name,
			accuracy=accuracy,
			precision=precision,
			recall=recall,
			f1_score=f1,
			confusion_matrix=cm,
			classification_report=cr,
			event_distribution=event_distribution,
			file_results=file_results
		)

	def safe_division(self, numerator, denominator):
		"""Безопасное деление"""
		return numerator / denominator if denominator > 0 else 0.0

	def print_detailed_analysis(self, result: DetailedTestResult):
		"""Детальный анализ результатов"""
		print(f"\n{'=' * 60}")
		print(f"ДЕТАЛЬНЫЙ АНАЛИЗ: {result.algorithm_name}")
		print(f"{'=' * 60}")

		print(f"\n📊 РАСПРЕДЕЛЕНИЕ СОБЫТИЙ:")
		dist = result.event_distribution
		print(f"   Всего samples: {dist['total_samples']}")
		print(f"   Истинные фиксации: {dist['true_fixations']} ({dist['fixation_ratio_true'] * 100:.1f}%)")
		print(f"   Предсказанные фиксации: {dist['predicted_fixations']} ({dist['fixation_ratio_pred'] * 100:.1f}%)")

		print(f"\n🎯 ОСНОВНЫЕ МЕТРИКИ:")
		print(f"   Accuracy:  {result.accuracy:.3f}")
		print(f"   Precision: {result.precision:.3f}")
		print(f"   Recall:    {result.recall:.3f}")
		print(f"   F1-Score:  {result.f1_score:.3f}")

		print(f"\n📈 МАТРИЦА ОШИБОК:")
		print(f"   [[TN={result.confusion_matrix[0, 0]} FP={result.confusion_matrix[0, 1]}]")
		print(f"    [FN={result.confusion_matrix[1, 0]} TP={result.confusion_matrix[1, 1]}]]")

		print(f"\n📋 CLASSIFICATION REPORT:")
		print(result.classification_report)

		# Анализ по файлам
		print(f"\n📁 РЕЗУЛЬТАТЫ ПО ФАЙЛАМ:")
		for filename, metrics in result.file_results.items():
			if metrics:
				print(f"   {filename}:")
				print(f"     Accuracy: {metrics['accuracy']:.3f}, "
				      f"Precision: {metrics['precision']:.3f}, "
				      f"Recall: {metrics['recall']:.3f}")
				print(f"     Истинные фиксации: {metrics['true_fixation_ratio'] * 100:.1f}%, "
				      f"Предсказанные: {metrics['prediction_stats']['fixation_ratio'] * 100:.1f}%")


def corrected_velocity_algorithm(x_coords: np.ndarray, y_coords: np.ndarray,
                                 velocity_threshold: float = None,
                                 use_adaptive: bool = False,
                                 sampling_rate: float = 500,
                                 **kwargs) -> np.ndarray:
	"""
	ИСПРАВЛЕННЫЙ алгоритм детекции фиксаций с правильными единицами
	"""
	# ВЫЧИСЛЕНИЕ СКОРОСТИ В ПРАВИЛЬНЫХ ЕДИНИЦАХ
	dx = np.diff(x_coords, prepend=x_coords[0])
	dy = np.diff(y_coords, prepend=y_coords[0])
	dt = 1.0 / sampling_rate  # 0.002 секунды для 500 Гц
	velocity = np.sqrt(dx ** 2 + dy ** 2) / dt  # пиксели/СЕКУНДУ!

	# Адаптивный порог (если не задан)
	if velocity_threshold is None or use_adaptive:
		# Берем 90-й перцентиль скорости в ПРАВИЛЬНЫХ единицах
		velocity_threshold = np.percentile(velocity, 90) * 0.5
		print(f"Адаптивный порог: {velocity_threshold:.1f} px/s")

	# Бинарная классификация
	predictions = (velocity < velocity_threshold).astype(int)

	return predictions


# ПРИМЕР ИСПОЛЬЗОВАНИЯ
if __name__ == "__main__":
	tester = AdvancedEyeMovementTester(
		r'C:\C++_projects\EyeMovementDetectorEvaluation\annotated_data\data used in the article'
	)

	print("Загрузка данных...")
	tester.load_all_datasets(max_files=3)

	# Теперь используем пороги в пикселях/СЕКУНДУ
	algorithms = {
		'IVT 30px/s': lambda x, y: corrected_velocity_algorithm(x, y, velocity_threshold=30),
		'IVT 100px/s': lambda x, y: corrected_velocity_algorithm(x, y, velocity_threshold=100),
		'IVT 200px/s': lambda x, y: corrected_velocity_algorithm(x, y, velocity_threshold=200),
		'Адаптивный': lambda x, y: corrected_velocity_algorithm(x, y, use_adaptive=True),
	}

	print("\n" + "=" * 60)
	print("ДЕТАЛЬНОЕ ТЕСТИРОВАНИЕ АЛГОРИТМОВ")
	print("=" * 60)

	for name, algo in algorithms.items():
		result = tester.test_algorithm_detailed(
			algorithm=algo,
			algorithm_name=name,
			target_events=[1]  # Фиксации
		)
		tester.print_detailed_analysis(result)