import glob
import re

import numpy as np

from ai.ai_features import (OTHER, FIO, DEPARTMENT, DATETIME, ACCESS, DOOR, KEY, ENTER, EXIT,
                            ROLES_COUNT, STRUCTURE_SIZE, GENDER_SIZE,
                            structure_features, gender_features)
from ai.ai_models import MODELS_DIR, STRUCTURE_FILE, GENDER_FILE
from ai.ai_names import random_fio
from ai.ai_network import Network
from csv_read import csv_read


"""
Обучение нейросетей: python -m ai.ai_train

Сеть разбора структуры учится на отчетах из каталога csv_in и на искусственных отчетах,
ответы для нее размечает "учитель" - набор правил, который нужен только при обучении
Сеть определения пола учится на ФИО, собранных из словаря ai_names
"""


companies = ["Законодательное Собрание ЯНАО", "Администрация города", "Аппарат Губернатора",
             "ООО Северный ветер", "Департамент финансов"]
department_heads = ["Отдел", "Управление", "Сектор", "Департамент", "Служба", "Комитет", "Группа"]
department_tails = ["кадров", "информационных технологий", "правового обеспечения", "делами",
                    "финансов и отчетности", "безопасности", "закупок", "протокола",
                    "документооборота", "контроля", "по связям с общественностью",
                    "охраны труда", "МТ и ЦО", "по работе с обращениями граждан"]
positions = ["Специалист", "главный специалист", "начальник отдела", "эксперт I категории",
             "консультант", "Помощник ", "депутат", "зам. начальника отдела", "водитель"]
accesses = ["Доступ предоставлен", "Доступ предоставлен", "Доступ предоставлен",
            "Доступ запрещен", "Проход", "Запрет прохода", "Доступ отклонен"]
doors = ["Турникет", "Турникет", "Турникет", "Дверь", "Калитка", "КПП-", "Шлагбаум", "Проходная"]
key_types = ["Proximity карта", "Proximity карта", "PIN-код", "Отпечаток пальца"]


def _pick(rng, items: list):
    return items[rng.integers(len(items))]


def _row(width: int, cells: dict) -> list:
    row = [""] * width
    for column, value in cells.items():
        row[column] = value
    return row


# Искусственный отчет: шапки сотрудников и события в обоих вариантах верстки страниц
def synthetic_report(rng, users: int) -> list:
    rows = []
    for _ in range(users):
        rows.append(_row(19, {2: "ФИО", 3: random_fio(rng, rng.random() < 0.5)}))
        rows.append(_row(19, {2: "Табельный номер", 3: "Дата рождения", 5: "ИНН",
                              8: "Рабочий телефон", 10: "Домашний телефон", 12: "Состояние"}))
        rows.append(_row(19, {2: str(rng.integers(1, 3000)), 12: "Активный"}))
        rows.append(_row(19, {2: "Компания", 6: "Подразделение", 12: "Должность", 16: "Комната"}))

        company = _pick(rng, companies)
        kind = rng.random()
        if kind < 0.15:
            department = company
        elif kind < 0.3:
            department = _pick(rng, positions).strip().capitalize() + " депутата"
        else:
            department = _pick(rng, department_heads) + " " + _pick(rng, department_tails)
        rows.append(_row(19, {2: company, 6: department, 12: _pick(rng, positions)}))

        key = "".join(_pick(rng, list("0123456789ABCDEF")) for _ in range(16))
        for _ in range(rng.integers(1, 9)):
            # Разрыв страницы меняет число колонок в строках событий
            wide = rng.random() < 0.6
            if rng.random() < 0.3:
                if wide:
                    rows.append(_row(19, {0: "Дата/время", 2: "Событие", 3: "Дверь", 6: "Зона доступа",
                                          9: "Ключ (идентификатор)", 14: "Комментарий"}))
                    rows.append(_row(19, {9: "тип ", 11: "код"}))
                else:
                    rows.append(_row(14, {0: "Дата/время", 2: "Событие", 3: "Дверь", 5: "Зона доступа",
                                          7: "Ключ (идентификатор)", 10: "Комментарий"}))
                    rows.append(_row(14, {7: "тип ", 8: "код"}))

            number = rng.integers(1, 10)
            door = _pick(rng, doors) + str(number)
            moment = (f"{rng.integers(1, 29):02}.{rng.integers(1, 13):02}.{rng.integers(2020, 2040)} "
                      f"{rng.integers(0, 24):02}:{rng.integers(0, 60):02}:{rng.integers(0, 60):02}")
            reader = rng.integers(1, 3)
            comment = (f"{number}: {'Вход' if reader == 1 else 'Выход'}   {door},   "
                       f"Считыватель {reader}, Прибор {number}")
            if rng.random() < 0.08:
                comment = ""
            event = {0: moment, 2: _pick(rng, accesses), 3: door}
            if wide:
                event.update({9: _pick(rng, key_types), 11: key, 14: comment})
            else:
                event.update({7: _pick(rng, key_types), 8: key, 10: comment})
            rows.append(_row(19 if wide else 14, event))

    return rows


# Учитель: разметка ролей ячеек отчета по правилам
def teacher(rows: list) -> dict:
    labels = {}
    department_column = None
    for row_id, row in enumerate(rows):
        filled = [column for column, value in enumerate(row) if value.strip()]
        if not filled:
            continue
        texts = [row[column].strip() for column in filled]

        if department_column is not None and department_column in filled:
            labels[(row_id, department_column)] = DEPARTMENT
        department_column = row.index("Подразделение") if "Подразделение" in row else None

        if "ФИО" in texts[:-1]:
            labels[(row_id, filled[texts.index("ФИО") + 1])] = FIO

        if re.fullmatch(r"\d\d\.\d\d\.\d{4} \d{1,2}:\d\d:\d\d", texts[0]):
            labels[(row_id, filled[0])] = DATETIME
            for order, (column, text) in enumerate(zip(filled, texts)):
                if order == 1:
                    labels[(row_id, column)] = ACCESS
                elif order == 2:
                    labels[(row_id, column)] = DOOR
                elif re.fullmatch(r"[0-9A-F]{12,}", text):
                    labels[(row_id, column)] = KEY
                elif "Выход" in text:
                    labels[(row_id, column)] = EXIT
                elif "Вход" in text:
                    labels[(row_id, column)] = ENTER

    return labels


def _split(rng, features: np.ndarray, answers: np.ndarray) -> tuple:
    order = rng.permutation(len(features))
    test = order[:len(order) // 10]
    learn = order[len(order) // 10:]
    return features[learn], answers[learn], features[test], answers[test]


def train_structure(rng) -> Network:
    rows = []
    for file_csv in sorted(glob.glob("csv_in/*.csv")):
        rows += csv_read(file_csv) or []
    rows += synthetic_report(rng, 400)

    features, cells = structure_features(rows)
    labels = teacher(rows)
    answers = np.array([labels.get(cell, OTHER) for cell in cells])

    x_learn, y_learn, x_test, y_test = _split(rng, features, answers)
    network = Network([STRUCTURE_SIZE, 48, ROLES_COUNT])
    network.fit(x_learn, y_learn, epochs=12)
    accuracy = (network.predict(x_test).argmax(axis=1) == y_test).mean()
    print(f"Сеть разбора структуры: ячеек {len(features)}, точность на проверке {accuracy:.4f}")
    return network


def train_gender(rng) -> Network:
    genders = rng.integers(0, 2, 16000)
    features = np.array([gender_features(random_fio(rng, bool(female))) for female in genders])

    x_learn, y_learn, x_test, y_test = _split(rng, features, genders)
    network = Network([GENDER_SIZE, 32, 2])
    network.fit(x_learn, y_learn, epochs=8)
    accuracy = (network.predict(x_test).argmax(axis=1) == y_test).mean()
    print(f"Сеть определения пола: ФИО {len(features)}, точность на проверке {accuracy:.4f}")
    return network


def train() -> None:
    rng = np.random.default_rng(2026)
    MODELS_DIR.mkdir(exist_ok=True)
    train_structure(rng).save(STRUCTURE_FILE)
    train_gender(rng).save(GENDER_FILE)


if __name__ == "__main__":
    train()
