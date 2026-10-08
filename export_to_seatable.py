import os
import re
from pathlib import Path

from seatable_api import Base


"""
Экспорт готового отчета в таблицу SeaTable

Настройки подключения берутся из переменных окружения или из файла .env в корне проекта:
SEATABLE_API_TOKEN, SEATABLE_SERVER_URL, SEATABLE_TABLE_NAME

Строки, которые уже есть в таблице, повторно не добавляются:
у них только заполняются пустые подразделение и пол
"""


ENV_FILE = Path(__file__).parent / ".env"

# Колонки, по которым событие отчета сравнивается со строкой таблицы
KEY_COLUMNS = ["Табельный номер", "Дата", "Время", "Доступ", "Дверь", "Событие"]
# Колонки, которые заполняются у уже существующих строк
FILL_COLUMNS = ["Подразделение", "Пол"]
# Сколько строк читается и записывается за одно обращение к серверу
BATCH_SIZE = 1000


# Настройки из файла .env, переменные окружения имеют приоритет
def _settings() -> dict:
    settings = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and not name.strip().startswith("#"):
                settings[name.strip()] = value.strip().strip("\"'")
    for name in ("SEATABLE_API_TOKEN", "SEATABLE_SERVER_URL", "SEATABLE_TABLE_NAME"):
        if os.environ.get(name):
            settings[name] = os.environ[name]
    return settings


# Строки таблицы за период отчета: ключ события -> список строк
def _existing_rows(base: Base, table_name: str, dates: list) -> dict:
    existing = {}
    if not dates:
        return existing

    columns = ", ".join(f"`{column}`" for column in KEY_COLUMNS + FILL_COLUMNS)
    offset = 0
    while True:
        rows = base.query(f"select _id, {columns} from `{table_name}` "
                          f"where `Дата` >= '{min(dates)}' and `Дата` <= '{max(dates)}' "
                          f"order by _id limit {BATCH_SIZE} offset {offset}")
        for row in rows:
            key = tuple(row.get(column) or "" for column in KEY_COLUMNS)
            existing.setdefault(key, []).append(row)
        if len(rows) < BATCH_SIZE:
            return existing
        offset += BATCH_SIZE


def export_to_seatable(list_final: list) -> str:
    settings = _settings()
    if not settings.get("SEATABLE_API_TOKEN"):
        return "Экспорт в SeaTable пропущен: не задан SEATABLE_API_TOKEN"
    table_name = settings.get("SEATABLE_TABLE_NAME", "Доступ в здание")

    try:
        base = Base(settings["SEATABLE_API_TOKEN"], settings.get("SEATABLE_SERVER_URL", "https://table.yanao.ru"))
        base.auth()

        events = [dict(zip(list_final[0], row)) for row in list_final[1:]]
        dates = sorted({event["Дата"] for event in events
                        if re.fullmatch(r"\d{4}\.\d\d\.\d\d", event["Дата"])})
        existing = _existing_rows(base, table_name, dates)

        rows_new = []
        rows_update = []
        for event in events:
            same = existing.get(tuple(event[column] for column in KEY_COLUMNS))
            if not same:
                rows_new.append(event)
                continue
            row = same.pop(0)
            fill = {column: event[column] for column in FILL_COLUMNS
                    if event[column] and not row.get(column)}
            if fill:
                rows_update.append({"row_id": row["_id"], "row": fill})

        for start in range(0, len(rows_new), BATCH_SIZE):
            base.batch_append_rows(table_name, rows_new[start:start + BATCH_SIZE])
        for start in range(0, len(rows_update), BATCH_SIZE):
            base.batch_update_rows(table_name, rows_update[start:start + BATCH_SIZE])

        return (f"Экспорт в SeaTable: добавлено строк - {len(rows_new)}, "
                f"дополнено - {len(rows_update)}, без изменений - "
                f"{len(events) - len(rows_new) - len(rows_update)}")
    except Exception as e:
        return f"Ошибка экспорта в SeaTable!\n {e}"
