import re


def extract_index(filename):
    # Удаляем расширение файла
    name_without_extension = filename.rsplit('.', 1)[0]
    # Находим все числа в имени
    numbers = re.findall(r'(\d+)', name_without_extension)
    if numbers:
        # Берем последнее число
        return int(numbers[-1])
    else:
        raise ValueError(f"Индекс не найден в имени файла: {filename}")

def sort_filenames(filenames):
    return sorted(filenames, key=extract_index)

def check_sorted(filenames):
    indices = [extract_index(name) for name in filenames]
    return indices == sorted(indices)