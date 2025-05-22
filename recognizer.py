import os
import base64
import requests
from requests import Response
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageOps
from dotenv import load_dotenv

load_dotenv()

class GeminiRecognizer:
    def __init__(self):
        self.proxy_url = os.getenv("PROXY_URL")
        self.proxy_secret = os.getenv("PROXY_SECRET")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.preprocessing_states = dict()

    def set_preprocessing(self, chat_id: int, state: bool):
        """Устанавливает состояние предобработки для конкретного чата"""
        self.preprocessing_states[chat_id] = state

    def get_preprocessing_state(self, chat_id: int) -> bool:
        """Возвращает текущее состояние предобработки"""
        return self.preprocessing_states.get(chat_id, True)  # По умолчанию включено

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Предобработка изображения для улучшения распознавания"""
        try:
            # Конвертация в оттенки серого
            img = image.convert('L')
            
            # Автоматическая коррекция контраста
            img = ImageOps.autocontrast(img)
            
            # Стабилизация яркости
            enhancer = ImageEnhance.Brightness(img)
            brightness_factor = self._calculate_brightness_factor(img)
            img = enhancer.enhance(brightness_factor)
            
            # Увеличение резкости
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(2.0)
            
            # Удаление шумов
            img = self._remove_noise(img)
            
            return img.convert("RGB")
            
        except Exception as e:
            print(f"Ошибка предобработки: {str(e)}")
            return image

    def _calculate_brightness_factor(self, image: Image.Image) -> float:
        """Рассчет коэффициента коррекции яркости"""
        histogram = image.histogram()
        pixels = sum(histogram)
        brightness = sum(i * num for i, num in enumerate(histogram)) / (pixels * 255)
        
        if brightness < 0.4:   # Слишком темное
            return 1.8 - brightness
        elif brightness > 0.6: # Слишком светлое
            return 0.6 / brightness
        return 1.0

    def _remove_noise(self, image: Image.Image, kernel_size=3) -> Image.Image:
        """Медианный фильтр для удаления шумов"""
        arr = np.array(image)
        padded = np.pad(arr, pad_width=kernel_size//2, mode='edge')
        result = np.zeros_like(arr)
        
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                result[i,j] = np.median(
                    padded[i:i+kernel_size, j:j+kernel_size]
                )
        return Image.fromarray(result)

    async def recognize_handwriting(self, image_data: BytesIO, chat_id: int) -> str:
        try:
            original_image = Image.open(image_data)
            
            # Применяем предобработку только если она включена
            if self.get_preprocessing_state(chat_id):
                processed_image = self._preprocess_image(original_image)
            else:
                processed_image = original_image.convert("RGB")
            
            buffered = BytesIO()
            processed_image.save(buffered, format="JPEG", quality=90)
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Формирование запроса
            payload = {
                "gemini_key": self.gemini_key,
                "model": "gemini-2.0-flash",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Распознай ВЕСЬ текст на изображении максимально точно. Сохрани оригинальную структуру текста. Ответ только текст без комментариев."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }]
            }

            headers = {
                "X-Proxy-Secret": self.proxy_secret,
                "Content-Type": "application/json"
            }

            # Отправка запроса
            response = requests.post(
                self.proxy_url,
                json=payload,
                headers=headers,
                timeout=15
            )

            if response.status_code != 200:
                return self._handle_error(response)

            return response.json()['choices'][0]['message']['content']

        except Exception as e:
            return f"Ошибка распознавания: {str(e)}"

    def _handle_error(self, response: Response) -> str:
        error_map = {
            400: "Некорректный запрос",
            401: "Ошибка авторизации прокси",
            403: "Неверный ключ Gemini",
            500: "Ошибка на сервере"
        }
        return error_map.get(response.status_code, "Неизвестная ошибка")