# -*- coding: utf-8 -*-

import os
from io import BytesIO
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from dotenv import load_dotenv
from recognizer import GeminiRecognizer
import asyncio

load_dotenv()

bot = Bot(token=os.getenv("BOT_TOKEN"))
dp = Dispatcher()
recognizer = GeminiRecognizer()

async def create_settings_keyboard(chat_id: int):
    """Создает инлайн-клавиатуру для настроек"""
    builder = InlineKeyboardBuilder()
    current_state = recognizer.get_preprocessing_state(chat_id)
    
    builder.button(
        text=f"Предобработка {'✅' if current_state else '❎'}",
        callback_data=f"toggle_preprocessing_{chat_id}"
    )
    return builder.as_markup()

@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    text = (
        "👋 *Привет!*\n"
        "Я бот, который может за секунду распознать рукописный/печатный текст на любом языке\n\n"
        "*Просто отправь фото 📸*\n\n"
        "*Кнопки ниже* используются для управления предобработкой\n\n"
        "*Если твое фото некачественное, включи предобработку* — это улучшит качество распознавания, "
        "но немного замедлит обработку\n"
        "Если с фото *всё ОК — можешь использовать бота без предобработки*"
    )

    await message.reply(
        text,
        parse_mode="Markdown",
        reply_markup=await create_settings_keyboard(message.chat.id)
    )

@dp.callback_query(F.data.startswith("toggle_preprocessing_"))
async def toggle_preprocessing(callback: types.CallbackQuery):
    chat_id = int(callback.data.split("_")[-1])
    new_state = not recognizer.get_preprocessing_state(chat_id)
    recognizer.set_preprocessing(chat_id, new_state)
    
    await callback.message.edit_reply_markup(
        reply_markup=await create_settings_keyboard(chat_id)
    )
    await callback.answer(f"Предобработка {'включена' if new_state else 'выключена'}")

@dp.message(F.photo | F.document)
async def handle_photo(message: types.Message):
    msg = await message.reply("🔄 Обрабатываю изображение...")
    
    try:
        # Скачивание фото
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        
        image_data = BytesIO()
        await bot.download_file(file.file_path, image_data)
        image_data.seek(0)

        # Распознавание с учетом настроек пользователя
        result = await recognizer.recognize_handwriting(
            image_data, message.chat.id
        )
        
        await msg.edit_text(
            f"📝 Результат:\n\n{result}",
            reply_markup=await create_settings_keyboard(message.chat.id)
        )

    except Exception as e:
        await msg.edit_text(f"⚠️ Ошибка: {str(e)}")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())