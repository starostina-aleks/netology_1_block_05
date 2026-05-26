#поиск Топ-3 моделей
from huggingface_hub import HfApi
import os
from dotenv import load_dotenv


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)
list_model_ids=[]
# Поиск моделей для генерации текста, отсортированных по скачиваниям
limit_model=10
print("Топ-10 моделей для token-classification:\n")
models = api.list_models(
    pipeline_tag="token-classification",
    #filter='russian',
    sort="downloads",
    limit=10
)
models_list = list(models)
print(len(models_list))
for model in models_list:
    print(f"  {model.id}")
    print(f"    Скачиваний: {model.downloads:,}")
    print(f"    Лайков: {model.likes:,}")
    print(f"    Теги: {', '.join(model.tags[:5]) if model.tags else 'нет'}")
    print()

    # Получаем детальную информацию о конкретной модели

    print("   Детали модели :\n")
    info = api.model_info(model.id)
    print(f"  ID: {info.id}")
    print(f"  Автор: {info.author}")
    print(f"  Скачиваний: {info.downloads:,}")
    print(f"  Лицензия: {info.card_data.get('license', 'не указана') if info.card_data else 'нет данных'}")
    print(f"  Теги: {', '.join(info.tags[:8]) if info.tags else 'нет'}")
    print(f"  Размер: {info.safetensors.total if info.safetensors else 'неизвестно'} параметров")
    list_model_ids.append(model.id)

EVAL_TASKS_NER=[
    {
        "text": "Сегодня в Санкт-Петербурге пройдет концерт группы Кино.",
        "entities": {
            "location": "Санкт-Петербург",
            "organization": "Кино"
        }
    },
    {
        "text": "Apple анонсировала новые iPhone в разгар осенней выставки.",
        "entities": {
            "organization": "Apple",
            "product": "iPhone"
        }
    },
    {
        "text": "Сборная Бразилии выиграла чемпионат мира по футболу в 2022 году.",
        "entities": {
            "organization": "Сборная Бразилии",
            "event": "чемпионат мира по футболу",
            "date": "2022"
        }
    },
    {
        "text": "Изучение языка Python открывает множество возможностей в IT.",
        "entities": {
            "language": "Python",
            "field": "IT"
        }
    },
    {
        "text": "Маша и Медведь - популярный мультфильм.",
        "entities": {
            "media": "Маша и Медведь",
            "genre": "мультфильм"
        }
    },
    {
        "text": "Новая книга Джоан Роулинг станет бестселлером.",
        "entities": {
            "person": "Джоан Роулинг",
            "product": "новая книга"
        }
    },
    {
        "text": "Кремль находится в центре Москвы.",
        "entities": {
            "location": "Москва",
            "landmark": "Кремль"
        }
    },
    {
        "text": "Том Хэнкс сыграл главную роль в фильме Форрест Гамп.",
        "entities": {
            "person": "Том Хэнкс",
            "media": "Форрест Гамп",
            "genre": "фильм"
        }
    },
    {
        "text": "Лучшие рестораны Парижа предлагают изысканные блюда.",
        "entities": {
            "location": "Париж",
            "category": "рестораны"
        }
    },
    {
        "text": "Microsoft выпустила обновление для Windows 11.",
        "entities": {
            "organization": "Microsoft",
            "product": "Windows 11"
        }
    },
    {
        "text": "Спортсмены из Китая выиграли золото на Олимпийских играх.",
        "entities": {
            "location": "Китай",
            "event": "Олимпийские игры"
        }
    },
    {
        "text": "Эйфелева башня является символом Парижа.",
        "entities": {
            "landmark": "Эйфелева башня",
            "location": "Париж"
        }
    },
    {
        "text": "День Победы празднуется 9 мая в России.",
        "entities": {
            "event": "День Победы",
            "date": "9 мая",
            "location": "Россия"
        }
    },
    {
        "text": "Литература Льва Толстого вдохновила множество поколений.",
        "entities": {
            "person": "Лев Толстой",
            "field": "литература"
        }
    },
    {
        "text": "Вулкан Везувий расположен в Италии.",
        "entities": {
            "landmark": "Вулкан Везувий",
            "location": "Италия"
        }
    },
    {
        "text": "Британская королева Елизавета II скончалась в 2022 году.",
        "entities": {
            "person": "Елизавета II",
            "date": "2022"
        }
    },
    {
        "text": "Гарри Поттер - культовая книга для молодежи.",
        "entities": {
            "media": "Гарри Поттер",
            "genre": "книга"
        }
    },
    {
        "text": "Солярис - знаменитый роман Станислава Лема.",
        "entities": {
            "media": "Солярис",
            "person": "Станислав Лем",
            "genre": "роман"
        }
    },
    {
        "text": "Глобальное потепление - серьезная проблема для планеты.",
        "entities": {
            "issue": "глобальное потепление",
            "category": "проблема"
        }
    }
]

import json
import time
import re
from transformers import pipeline

results=[]
# Проход по моделям
for model in list_model_ids:
    print("=============================================================================================")
    print(f"Пробую загрузить модель: {model}")

    try:
        ner = pipeline(
            "token-classification",
            model=model,
            aggregation_strategy="simple",
            token=HF_TOKEN,
        )

        print(f"Модель загружена: {model}")
        results=[]
        for entry in EVAL_TASKS_NER:
            article = entry['text']
            # Измерение времени
            start_time = time.time()
            entities = ner(article)
            elapsed_time = time.time() - start_time
            list_ent=[]
            for ent in entities:
                 list_ent.append(f"  {ent['word']:20s} → {ent['entity_group']:5s} ({ent['score']:.0%})")
            # Запись результата
            results.append({
            "text": article,
            "ner_model": list_ent,
            "ner_test": entry['entities'],
            "elapsed_time": elapsed_time
            })
        # Убираем лишние символы из названия модели
        model_name_cleaned = re.sub(r"[^a-zA-Z0-9_]", "", model)  # Оставляем только буквы, цифры и символы подчеркивания
        model_results_file = f"ner_{model_name_cleaned}_results.json"

        print(model_results_file)


        # Сохранение результатов в JSON файл для каждой модели
        with open(model_results_file, "w", encoding="utf-8") as f:
           json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"Результаты для модели {model} сохранены в {model_results_file}")
    except Exception as e:
        print(f"Ошибка загрузки модели {model}: {e}. Переход к следующей модели.")