import csv
from datetime import datetime


def ai_writer_csv(list_source: list) -> str:
    file_csv = datetime.now().strftime("%Y%m%d-%H%M%S.csv")

    try:
        with open(f"csv_out/{file_csv}", 'w', encoding="windows-1251", newline='') as file_obj:
            writer_obj = csv.writer(file_obj, lineterminator='\n')
            for row in list_source:
                writer_obj.writerow(row)

        return f"Файл {file_csv} успешно создан!"
    except Exception as e:
        return f"Возникла непредвиденная ошибка!\n {e}"
