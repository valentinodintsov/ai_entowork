import re
import zlib

import numpy as np


"""
Преобразование данных в признаки для нейросетей

structure_features - признаки каждой непустой ячейки отчета
gender_features    - признаки ФИО сотрудника
"""


# Роли ячеек отчета (выходы сети разбора структуры)
OTHER, FIO, DEPARTMENT, DATETIME, ACCESS, DOOR, KEY, ENTER, EXIT = range(9)
ROLES_COUNT = 9

# Пол сотрудника (выходы сети определения пола)
GENDERS = ["М", "Ж"]

OWN_SIZE = 256
LEFT_SIZE = 64
ABOVE_SIZE = 64
ROW_SIZE = 128
NUMBERS_SIZE = 13
STRUCTURE_SIZE = OWN_SIZE + LEFT_SIZE + ABOVE_SIZE + ROW_SIZE + NUMBERS_SIZE

SUFFIX_SIZE = 256
NAME_SIZE = 256
GENDER_SIZE = SUFFIX_SIZE * 3 + NAME_SIZE


def _bucket(text: str, size: int) -> int:
    return zlib.crc32(text.encode("utf-8")) % size


# Слова ячейки в нижнем регистре, любое число заменяется на "0"
def _words(text: str) -> list:
    words = re.findall(r"[a-zа-я]+|\d+", text.lower().replace("ё", "е"))
    return ["0" if word[0].isdigit() else word for word in words]


def _numbers(text: str, column: int, order: int, filled: int, width: int) -> list:
    letters = [char for char in text if char.isalpha()]
    return [
        min(len(text), 40) / 40,
        min(len(text.split()), 8) / 8,
        sum(char.isdigit() for char in text) / len(text),
        len(letters) / len(text),
        sum(char.isupper() for char in letters) / len(letters) if letters else 0,
        sum(char in "0123456789ABCDEF" for char in text) / len(text),
        min(text.count(":"), 4) / 4,
        min(text.count("."), 4) / 4,
        min(text.count(","), 4) / 4,
        min(column, 20) / 20,
        min(order, 8) / 8,
        min(filled, 8) / 8,
        min(width, 20) / 20,
    ]


def structure_features(rows: list) -> tuple:
    """
    Признаки непустых ячеек: слова самой ячейки, соседа слева, ячейки над ней
    и всей строки, плюс числовые характеристики текста и положения
    Возвращает матрицу признаков и список координат ячеек (строка, колонка)
    """
    features = []
    cells = []
    previous = []

    for row_id, row in enumerate(rows):
        filled = [column for column, value in enumerate(row) if value.strip()]
        if not filled:
            continue

        row_words = {word for column in filled for word in _words(row[column])}
        left = ""
        for order, column in enumerate(filled):
            text = row[column].strip()
            above = previous[column] if column < len(previous) else ""
            vector = np.zeros(STRUCTURE_SIZE, dtype=np.float32)

            offset = 0
            for word in _words(text):
                vector[offset + _bucket(word, OWN_SIZE)] = 1
            offset += OWN_SIZE
            for word in _words(left):
                vector[offset + _bucket(word, LEFT_SIZE)] = 1
            offset += LEFT_SIZE
            for word in _words(above):
                vector[offset + _bucket(word, ABOVE_SIZE)] = 1
            offset += ABOVE_SIZE
            for word in row_words:
                vector[offset + _bucket(word, ROW_SIZE)] = 1
            offset += ROW_SIZE
            vector[offset:] = _numbers(text, column, order, len(filled), len(row))

            features.append(vector)
            cells.append((row_id, column))
            left = text
        previous = row

    if not features:
        return np.zeros((0, STRUCTURE_SIZE), dtype=np.float32), cells
    return np.array(features), cells


def gender_features(fio: str) -> np.ndarray:
    """
    Признаки ФИО: окончания (1-4 буквы) фамилии, имени и отчества, а также имя целиком
    """
    vector = np.zeros(GENDER_SIZE, dtype=np.float32)
    words = fio.lower().replace("ё", "е").split()
    parts = [words[0] if words else "",
             words[1] if len(words) > 1 else "",
             " ".join(words[2:])]

    for part_id, part in enumerate(parts):
        for length in range(1, min(len(part), 4) + 1):
            vector[part_id * SUFFIX_SIZE + _bucket(f"{length}:{part[-length:]}", SUFFIX_SIZE)] = 1
    if parts[1]:
        vector[SUFFIX_SIZE * 3 + _bucket(parts[1], NAME_SIZE)] = 1

    return vector
