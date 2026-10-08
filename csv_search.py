import os
import glob


"""
Поиск файлов по маске и их сортировка по дате создания
Вывод последнего созданного файла
Если файлов нет, возвращается None
"""


def csv_search():
    files = glob.glob("csv_in/*.csv")
    if not files:
        return None
    files.sort(key=os.path.getctime)
    return files[-1]
