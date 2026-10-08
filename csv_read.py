import csv


"""
Чтение данных из файла
Копирование данных в глобальную переменную
"""


def csv_read(file_csv):
    list_from_file_csv = []

    try:
        with open(file_csv, encoding='windows-1251', newline='') as file_obj:
            reader_obj = csv.reader(file_obj, delimiter=';')
            for row in reader_obj:
                if len(row) != 0:
                    list_from_file_csv.append(row)

        return list_from_file_csv
    except Exception as e:
        print(f"Непредвиденная ошибка чтения файла!\b{e}")
        return False
