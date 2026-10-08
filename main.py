import sys

from csv_read import csv_read
from csv_search import csv_search
from ai.ai_main import ai_main
from export_to_seatable import export_to_seatable


# Находим последний csv-файл
file_name = csv_search()
if file_name is None:
    sys.exit("В каталоге csv_in нет csv-файла. Загрузите отчет и запустите программу снова.")

# Считываем данные из файла
list_source = csv_read(file_name)
if not list_source:
    sys.exit(f"Файл {file_name} пуст или не прочитан. Программа остановлена.")

# Нейросети разбирают данные, определяют пол сотрудников и формируют готовый csv-файл
list_final = ai_main(list_source)

# Экспортируем готовые данные в SeaTable
print(export_to_seatable(list_final))
