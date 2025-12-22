"""
Пример использования скрипта валидации алгоритмов сегментации

Этот скрипт демонстрирует, как использовать validate_segmentation.py
для оценки точности алгоритмов сегментации движений глаз.
"""

import subprocess
import sys
from pathlib import Path

# Пути к файлам (настройте под ваши данные)
PUPIL_DATA = "моргания.csv"  # Файл с данными о зрачках
ANNOTATIONS = "моргания_annotations.csv"  # Файл с разметкой
OUTPUT = "segmentation_validation_results.csv"  # Файл для сохранения результатов

# Путь к скрипту валидации
VALIDATION_SCRIPT = Path(__file__).parent.parent / "src" / "utils" / "validate_segmentation.py"


def main():
    """Запуск валидации алгоритмов сегментации."""
    
    # Проверка существования файлов
    pupil_path = Path(PUPIL_DATA)
    annotations_path = Path(ANNOTATIONS)
    
    if not pupil_path.exists():
        print(f"Ошибка: Файл с данными о зрачках не найден: {pupil_path}")
        print("Укажите правильный путь к файлу в переменной PUPIL_DATA")
        return 1
    
    if not annotations_path.exists():
        print(f"Ошибка: Файл с разметкой не найден: {annotations_path}")
        print("Укажите правильный путь к файлу в переменной ANNOTATIONS")
        return 1
    
    if not VALIDATION_SCRIPT.exists():
        print(f"Ошибка: Скрипт валидации не найден: {VALIDATION_SCRIPT}")
        return 1
    
    # Формирование команды
    cmd = [
        sys.executable,
        str(VALIDATION_SCRIPT),
        "--pupil_data", str(pupil_path),
        "--annotations", str(annotations_path),
        "--output", OUTPUT,
        # Можно настроить параметры алгоритмов:
        # "--ivt_threshold", "110",
        # "--idt_threshold", "7",
        # "--fps", "30",
        # "--min_fixation_duration", "0.1",
        # "--iou_threshold", "0.1"
    ]
    
    print("Запуск валидации алгоритмов сегментации...")
    print(f"  Данные о зрачках: {pupil_path}")
    print(f"  Разметка: {annotations_path}")
    print(f"  Результаты будут сохранены в: {OUTPUT}")
    print()
    
    # Запуск скрипта
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\nВалидация завершена успешно!")
        print(f"Результаты сохранены в: {OUTPUT}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nОшибка при выполнении валидации: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nВалидация прервана пользователем")
        return 1


if __name__ == "__main__":
    sys.exit(main())

