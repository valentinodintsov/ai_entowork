from pathlib import Path

import numpy as np

from ai.ai_features import (FIO, DEPARTMENT, DATETIME, ACCESS, DOOR, KEY, ENTER, EXIT, GENDERS,
                            structure_features, gender_features)
from ai.ai_network import Network


"""
Применение обученных нейросетей

ai_structure - разбор отчета: сеть определяет роль каждой ячейки
ai_gender    - определение пола сотрудника по ФИО
"""


MODELS_DIR = Path(__file__).parent / "models"
STRUCTURE_FILE = MODELS_DIR / "structure.npz"
GENDER_FILE = MODELS_DIR / "gender.npz"


# Загрузка весов сетей, при их отсутствии сети обучаются заново
def load_networks() -> tuple:
    if not STRUCTURE_FILE.exists() or not GENDER_FILE.exists():
        from ai.ai_train import train
        train()
    return Network.load(STRUCTURE_FILE), Network.load(GENDER_FILE)


def ai_structure(network: Network, rows: list) -> list:
    """
    Возвращает события в виде
    [код ключа, ФИО, дата, время, доступ, дверь, событие, подразделение]
    """
    features, cells = structure_features(rows)
    if not cells:
        return []
    probabilities = network.predict(features)
    roles = probabilities.argmax(axis=1)

    # Для каждой строки отчета: роль -> текст ячейки с наибольшей уверенностью сети
    found = {}
    for (row_id, column), role, probability in zip(cells, roles, probabilities.max(axis=1)):
        row_roles = found.setdefault(row_id, {})
        if role not in row_roles or probability > row_roles[role][0]:
            row_roles[role] = (probability, " ".join(rows[row_id][column].split()))

    events = []
    user_fio = ""
    user_department = ""
    for row_id in sorted(found):
        row_roles = {role: text for role, (_, text) in found[row_id].items()}
        if FIO in row_roles:
            user_fio = row_roles[FIO]
            user_department = ""
        if DEPARTMENT in row_roles:
            user_department = row_roles[DEPARTMENT]
            # Подразделение стоит ниже событий только при разрыве страницы внутри шапки сотрудника
            for event in reversed(events):
                if event[1] != user_fio or event[7]:
                    break
                event[7] = user_department
        if DATETIME in row_roles:
            user_date, _, user_time = row_roles[DATETIME].partition(" ")
            user_date = ".".join(reversed(user_date.split(".")))
            user_event = "Вход" if ENTER in row_roles else "Выход" if EXIT in row_roles else ""
            events.append([row_roles.get(KEY, ""), user_fio, user_date, user_time.strip(),
                           row_roles.get(ACCESS, ""), row_roles.get(DOOR, ""), user_event,
                           user_department])

    return events


def ai_gender(network: Network, fio_list: list) -> dict:
    """
    Возвращает словарь ФИО -> (пол, уверенность сети)
    """
    if not fio_list:
        return {}
    probabilities = network.predict(np.array([gender_features(fio) for fio in fio_list]))
    return {fio: (GENDERS[row.argmax()], float(row.max()))
            for fio, row in zip(fio_list, probabilities)}
