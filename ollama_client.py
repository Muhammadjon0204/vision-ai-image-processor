"""
ollama_client.py — обёртка над LM Studio (OpenAI-совместимый API)

Запуск LM Studio:
  1. Открыть LM Studio → вкладка "Local Server" (или Developer)
  2. Загрузить модель (Llama 3.1 8B Q4_K_M рекомендуется)
  3. Нажать "Start Server" → сервер поднимется на http://localhost:1234
"""
import requests

# LM Studio поднимает OpenAI-совместимый сервер на порту 1234
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

# Имя модели — можно любое, LM Studio игнорирует это поле
# и использует загруженную в GUI модель
MODEL = "local-model"


def ask_ollama(prompt: str) -> str:
    """
    Отправляет запрос в LM Studio через OpenAI Chat Completions API.
    Функция называется ask_ollama для совместимости с остальным кодом.
    """
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,   # низкая — для стабильного JSON-вывода
        "max_tokens": 200,
        "stream": False
    }

    try:
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "LM Studio не запущен или сервер не стартован. "
            "Откройте LM Studio → вкладка 'Local Server' → 'Start Server'"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "LM Studio не ответил за 120 секунд. "
            "Модель ещё загружается или слишком тяжёлая для данного железа."
        )
    except KeyError:
        raise RuntimeError(f"Неожиданный формат ответа LM Studio: {response.text[:300]}")
    except Exception as e:
        raise RuntimeError(f"Ошибка LM Studio: {e}")
