# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 20:42:29 2024

@author: getma
"""

import cv2
import numpy as np
import os

def remove_small_objects(image, min_size):
    # Преобразование изображения в бинарное (если оно не бинарное)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Найти все компоненты связности
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # Создать пустое изображение для маски
    mask = np.zeros(binary_image.shape, dtype=np.uint8)

    # Обойти все компоненты связности и заполнить их на маске, если они больше min_size
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            mask[labels == i] = 255

    # Применить маску к исходному изображению
    result_image = cv2.bitwise_and(image, image, mask=mask)

    return result_image

# Пример использования
if __name__ == "__main__":
    input_folder =  "result_t_dataset_100_slices_1_classes/mitochondria"  # Папка с исходными изображениями
    output_folder = "result_t_dataset_100_slices_1_classes/new_mitohondria"  # Папка для сохранения обработанных изображений
    min_size = 900  # Минимальный размер пятен для удаления

    # Убедитесь, что папка для вывода существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Обработка всех изображений в папке
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Полный путь к файлу
            input_path = os.path.join(input_folder, filename)
            
            # Загрузка изображения
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            
            # Удаление пятен меньше заданного размера
            cleaned_image = remove_small_objects(image, min_size)
            
            # Сохранение результата
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cleaned_image)
            
            print(f"Обработано: {filename}")

    print("Все изображения обработаны и сохранены.")
