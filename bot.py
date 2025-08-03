import logging
import asyncio
import json
import requests
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton,FSInputFile
from aiogram.filters import Command
from docx import Document
import PyPDF2
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import random
import os
from dotenv import load_dotenv
from aiogram.client.session.aiohttp import AiohttpSession
import aiohttp


load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")

session = AiohttpSession(timeout=60)

bot = Bot(token=API_TOKEN, session=session)

LLM_URL = "https://api.intelligence.io.solutions/api/v1/chat/completions"
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
logging.basicConfig(level=logging.INFO)

dp = Dispatcher()

user_models = {}
user_tests = {}
user_answers = {}
user_original_texts = {}

@dp.message(CommandStart())
async def cmd_start(message: Message):
    photo_path = "lama.jpeg"
    photo = FSInputFile(photo_path)

    caption = (f"""
Привет, {message.from_user.full_name}! 👋

🤖 Я помогу тебе сгенерировать тест по любому учебному тексту.

🔍 Для наилучшего понимания и анализа текста используется модель LLaMA-4 — она показала себя наиболее точно в решении этой задачи.

📂 Отправь текст или файл (.txt, .pdf, .docx), чтобы начать!""".strip())
    await message.answer_photo(photo=photo, caption=caption)


@dp.message(F.document)
async def handle_document(message: Message):
    user_id = message.from_user.id
    model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

    file = await bot.download(message.document)
    filename = message.document.file_name.lower()

    if filename.endswith(".txt"):
        text = file.read().decode("utf-8", errors="ignore")
    elif filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif filename.endswith(".docx"):
        doc = Document(file)
        text = "\n".join(p.text for p in doc.paragraphs)
    elif message.document.file_size > 10 * 1024 * 1024:  # 10 MB
        await message.answer("⚠️ Файл слишком большой. Пожалуйста, отправь файл размером до 10MB.")
        return

    else:
        await message.answer("❌ Формат файла не поддерживается. Пожалуйста, загрузи .txt, .pdf или .docx")
        return

    # Сохраняем исходный текст для возможности заменить вопросы
    user_original_texts[user_id] = text

    prompt = generate_prompt(text)
    response_text = await make_neuro_request(prompt, model)
    quiz = parse_quiz(response_text)

    if not quiz:
        await message.answer("Не удалось распознать тест. Попробуйте другой текст или модель.")
        return

    user_tests[user_id] = quiz
    await message.answer("📘 Тест сгенерирован! Напиши /quiz, чтобы пройти его.")


@dp.message(F.text & ~F.text.startswith("/"))
async def handle_text(message: Message):
    user_id = message.from_user.id
    text = message.text

    if text.startswith("/"):
        # Игнорируем команды здесь, чтобы не конфликтовать с отдельными обработчиками
        return

    model = user_models.get(user_id, "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
    prompt = generate_prompt(text)
    response_text = await make_neuro_request(prompt, model)
    quiz = parse_quiz(response_text)
    if not quiz:
        await message.answer("Не удалось распознать тест. Возможно, модель вернула некорректный ответ.")
        logging.error(f"Невалидный JSON: {response_text}")
        return

    user_tests[user_id] = quiz
    user_original_texts[user_id] = text
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔁 Заменить вопросы", callback_data="replace_quiz")]
    ])
    await message.answer("Тест сгенерирован! Напишите /quiz, чтобы пройти его.")
    await message.answer("Не устраивают вопросы? Попробуй заменить:", reply_markup=keyboard)


@dp.message(Command("quiz"))
async def quiz_handler(message: Message):
    user_id = message.from_user.id
    logging.info(f"[DEBUG] Вызван /quiz от {user_id}")

    if user_id not in user_tests:
        logging.warning(f"[DEBUG] Нет теста для пользователя {user_id}")
        await message.answer("Сначала отправь учебный текст, чтобы сгенерировать тест.")
        return

    user_answers[user_id] = {"current": 0, "score": 0}
    await send_question(message.chat.id, user_id)


@dp.callback_query(lambda c: "|" in c.data)
async def handle_answer(callback: types.CallbackQuery):
    user_id = callback.from_user.id

    if user_id not in user_answers or user_id not in user_tests:
        await callback.answer("Сначала начните тест с /quiz", show_alert=True)
        return

    try:
        selected_letter, question_idx = callback.data.split("|")
        question_idx = int(question_idx)
    except Exception:
        await callback.answer("Некорректный ответ", show_alert=True)
        return

    quiz = user_tests[user_id]
    question = quiz[question_idx]
    correct = question["correct"]
    explanation = question.get("explanation", "❔ Объяснение недоступно.")

    is_correct = selected_letter == correct
    if is_correct:
        user_answers[user_id]["score"] += 1
        result = "✅ Правильно!"
    else:
        result = f"❌ Неправильно. Правильный ответ: {correct}) {question['options'][correct]}"

    user_answers[user_id]["current"] += 1

    # Собираем текст с объяснением
    full_text = (
        f"📘 Вопрос {question_idx + 1}: {question['question']}\n\n"
        f"📝 Вы выбрали: {selected_letter}) {question['options'].get(selected_letter, '—')}\n"
        f"{result}\n\n"
        f"💡 Объяснение: {explanation}"
    )

    # Удаляем клавиатуру, чтобы нельзя было отвечать повторно
    await callback.message.edit_text(full_text, reply_markup=None)

    # Переход к следующему вопросу
    await send_question(callback.message.chat.id, user_id)


@dp.callback_query(lambda c: c.data == "replace_quiz")
async def handle_replace_quiz(callback: types.CallbackQuery):
    user_id = callback.from_user.id

    if user_id not in user_tests:
        await callback.answer("Сначала отправь учебный текст, чтобы сгенерировать тест.", show_alert=True)
        return

    original_text = user_original_texts.get(user_id)
    if not original_text:
        await callback.answer("Исходный текст не найден.", show_alert=True)
        return

    await callback.message.edit_text("⏳ Генерирую новые вопросы...")

    prompt = generate_prompt(original_text)
    model = user_models.get(user_id, "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
    response_text = await make_neuro_request(prompt, model)
    logging.info(f"Ответ LLM при замене вопросов: {response_text[:300]}")

    quiz = parse_quiz(response_text)

    if not quiz:
        await callback.message.edit_text("❌ Не удалось сгенерировать новые вопросы. Попробуйте сократить текст.")
        await callback.answer()
        return

    user_tests[user_id] = quiz
    await callback.message.edit_text("✅ Вопросы обновлены. Напишите /quiz, чтобы пройти новый тест.")
    await callback.answer()


def generate_prompt(text: str) -> str:
    return f"""
Ты — преподаватель, создающий тесты для студентов.
Проанализируй следующий учебный текст и сгенерируй по нему 5 вопросов в формате множественного выбора.
Условия:
- Каждый вопрос должен быть по содержанию текста.
- У каждого вопроса 4 варианта ответа: A, B, C, D.
- Только один правильный вариант.
- Для каждого вопроса укажи краткое объяснение правильного ответа.
Формат ответа:
Ответ должен быть **только** валидным **JSON-массивом**, без заголовков, комментариев, пояснений, markdown-блоков или обёрток. **Никакого текста до или после JSON**. Строго следуй этому шаблону:

[
  {{
    "question": "текст вопроса",
    "options": {{
      "A": "текст варианта A",
      "B": "текст варианта B",
      "C": "текст варианта C",
      "D": "текст варианта D"
    }},
    "correct": "A/B/C/D",
    "explanation": "пояснение к правильному ответу"
  }},
  ...
]
Учебный текст:
{text}
"""


async def make_neuro_request(prompt: str, model: str) -> str:
    headers = {
        "Authorization": AUTH_TOKEN,
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = await asyncio.to_thread(
            lambda: requests.post(LLM_URL, headers=headers, json=data, timeout=60)
        )

        if response.status_code != 200:
            logging.error(f"[LLM ERROR] Статус {response.status_code}: {response.text}")
            return ""

        response_json = response.json()

        if "choices" not in response_json:
            logging.error(f"[LLM ERROR] Нет поля 'choices'. Ответ: {response_json}")
            return ""

        return response_json["choices"][0]["message"]["content"]

    except Exception as e:
        logging.exception("Ошибка при запросе к LLM")
        return ""


def parse_quiz(text: str):
    import logging
    logging.info(f"Получен ответ для парсинга (первые 300 символов): {text[:300]}")

    text = text.strip()

    # Убираем обёртки ```
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    # Убираем лишние скобки от LLM
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()

    if not text.strip().endswith("]"):
        logging.error("Ответ JSON обрывается — попробуйте сократить текст или увеличить таймаут.")
        logging.error(f"Невалидный JSON: {text}")
        return None

    try:
        quiz = json.loads(text)
        formatted_quiz = []
        for q in quiz:
            formatted_quiz.append({
                "question": q["question"],
                "options": q["options"],
                "correct": q["correct"].upper(),
                "explanation": q.get("explanation", "❔ Объяснение отсутствует.")
            })
        return formatted_quiz

    except Exception as e:
        logging.error(f"Ошибка парсинга JSON: {e}")
        logging.error(f"Невалидный JSON: {text}")
        return None


async def send_question(chat_id: int, user_id: int):
    try:
        quiz = user_tests[user_id]
        state = user_answers.setdefault(user_id, {"current": 0, "score": 0})
        idx = state["current"]

        if idx >= len(quiz):
            score = state["score"]
            result_text = get_result_message(score, len(quiz))
            await bot.send_message(chat_id, result_text.strip())



            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔁 Заменить вопросы", callback_data="replace_quiz")]
            ])
            await bot.send_message(chat_id, "Хотите сгенерировать новые вопросы по тому же тексту?",
                                   reply_markup=keyboard)
            await bot.send_message(chat_id, "📚 Хочешь ещё один тест? Отправь новый учебный текст или файл.")

            user_answers.pop(user_id, None)
            return

        q = quiz[idx]
        options = q["options"]

        # Формируем текст с вариантами
        text = f"Вопрос {idx + 1}: {q['question']}\n\n"
        for key, val in options.items():
            text += f"{key}) {val}\n"

        # Кнопки только с буквами (A, B, C, D)
        buttons = [
            InlineKeyboardButton(text=key, callback_data=f"{key}|{idx}")
            for key in options.keys()
        ]
        markup = InlineKeyboardMarkup(inline_keyboard=[buttons])

        await bot.send_message(chat_id, text.strip(), reply_markup=markup)

    except Exception as e:
        logging.exception(f"[ERROR] Ошибка при отправке вопроса: {e}")


def get_result_message(score: int, total: int) -> str:
    percent = (score / total) * 100

    if percent == 100:
        quotes = [
            "«Познай самого себя, и ты познаешь вселенную и богов.» — Сократ",
            "«Человек есть то, что он делает из самого себя.» — Жан-Поль Сартр",
        ]
    elif percent >= 80:
        quotes = [
            "«Учиться — это значит открывать то, что уже известно каждому.» — Ричард Бах",
            "«Терпение — это горький корень, дающий сладкие плоды.» — Жан-Жак Руссо",
        ]
    elif percent >= 50:
        quotes = [
            "«Ошибки — двери к открытию.» — Джеймс Джойс",
            "«Чем больше я узнаю, тем больше понимаю, как мало знаю.» — Сократ",
        ]
    else:
        quotes = [
            "«Неудача — это просто возможность начать снова, но уже более мудро.» — Генри Форд",
            "«Даже путь в тысячу ли начинается с первого шага.» — Лао-цзы",
        ]

    quote = random.choice(quotes)

    return f"""
✅ Тест завершён!
Правильных ответов: {score} из {total} ({int(percent)}%)

🧠 {quote}
"""



async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

