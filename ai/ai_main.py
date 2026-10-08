from ai.ai_models import load_networks, ai_structure, ai_gender
from ai.ai_writer_csv import ai_writer_csv


# Уверенность сети, ниже которой пол сотрудника стоит проверить вручную
GENDER_CONFIDENCE = 0.9


def ai_main(list_source: list) -> list:
    structure_network, gender_network = load_networks()

    # Сеть разбирает отчет на события
    events = ai_structure(structure_network, list_source)

    # Сеть определяет пол каждого сотрудника
    genders = ai_gender(gender_network, sorted({event[1] for event in events if event[1]}))

    list_final = [["Табельный номер", "ФИО", "Дата", "Время", "Доступ", "Дверь", "Событие",
                   "Подразделение", "Пол"]]
    for event in events:
        list_final.append(event + [genders[event[1]][0] if event[1] else ""])

    print(f"Сотрудников: {len(genders)}, событий: {len(events)}")
    for fio, (gender, confidence) in genders.items():
        if confidence < GENDER_CONFIDENCE:
            print(f"Проверьте пол: {fio} - {gender} (уверенность {confidence:.0%})")

    print(ai_writer_csv(list_final))
    return list_final
