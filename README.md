# Telegram Text Recognition Bot 🤖

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

Телеграм-бот для распознавания рукописного и печатного текста на изображениях с использованием Google Gemini API. Включает предобработку изображений для улучшения качества распознавания.

### Предварительные требования
- Docker и Docker Compose
- Telegram API Token ([@BotFather](https://t.me/BotFather))
- Google Gemini API Key ([AI Studio](https://aistudio.google.com/))
- Proxy для Gemini ([Vercel](https://github.com/uwulakai/vercel_proxy))

### Использование

1) ``/start`` - начало работы с ботом
2) Настройки через меню ``[🔄 Предобработка ✅]``

### Технологический стек
#### Основные компоненты
- Google Gemini API - распознавание текста
- Aiogram 3 - телеграм-бот
- Pillow - обработка изображений
- NumPy - матричные операции
#### Инфраструктура
- Docker - Контейнеризация
- [Vercel](https://github.com/uwulakai/vercel_proxy) - прокси 