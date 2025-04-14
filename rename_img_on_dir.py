import os
import re

def rename_images(directory, prefix="blizko"):
    """
    Переименовывает все файлы изображений в заданной директории в "prefix#",
    где # - порядковый номер файла.

    Args:
        directory: Путь к директории с изображениями.
        prefix: Префикс для новых имен файлов (по умолчанию "test").
    """

    # Регулярное выражение для поиска файлов изображений (можно расширить список)
    image_extensions = r"\.(jpg|jpeg|png|gif|bmp)$"
    pattern = re.compile(image_extensions, re.IGNORECASE)

    # Счетчик файлов
    count = 1

    # Обходим все файлы в директории
    for filename in os.listdir(directory):
        if pattern.search(filename):  # Проверяем, является ли файл изображением
            old_path = os.path.join(directory, filename)
            name, extension = os.path.splitext(filename)
            new_filename = f"{prefix}{count}{extension}"
            new_path = os.path.join(directory, new_filename)

            try:
                os.rename(old_path, new_path)
                print(f"Переименовано: {filename} -> {new_filename}")
                count += 1
            except OSError as e:
                print(f"Ошибка при переименовании {filename}: {e}")

if __name__ == "__main__":
    directory_path = "test_data/foto/blizko"
    rename_images(directory_path)
    print("Готово!")