# 👁 Vision AI — Image Processing

Система управления изображениями через естественный язык.  
LLM (LM Studio) парсит команду → OpenCV выполняет операцию → результат в браузере.

## Архитектура

```
Пользователь (браузер)
        ↓
   Flask app.py
        ↓
command_parser.py  →  LM Studio API (llama3.1-8B-Instruct Q4_K_M)
        ↓                    ↓
   JSON-команда        Prompt Engineering
        ↓
image_processor.py  →  OpenCV (19 операций)
        ↓
  base64 PNG  →  Web UI (светлая тема, режим сравнения)
```

## Модули

| Файл | Назначение |
|------|-----------|
| `app.py` | Flask-сервер, маршруты `/process` и `/health`, валидация, логирование |
| `command_parser.py` | Prompt engineering, парсинг ответа LLM → JSON |
| `image_processor.py` | 19 OpenCV-операций |
| `ollama_client.py` | HTTP-клиент к LM Studio (OpenAI-совместимый API) |
| `static/index.html` | Web UI: светлая тема, drag-and-drop, режим сравнения, история |
| `requirements.txt` | Python-зависимости |

## Установка

### 1. LM Studio (локальная LLM)

1. Скачать: [lmstudio.ai](https://lmstudio.ai)
2. В поиске найти и скачать модель: `Meta-Llama-3.1-8B-Instruct-Q4_K_M` (~4.92 GB)
3. Перейти на вкладку **Local Server** → выбрать модель → нажать **Start Server**
4. Сервер поднимется на `http://localhost:1234`

### 2. Python-зависимости

```bash
pip install -r requirements.txt
```

### 3. Запуск

```bash
python app.py
# Открыть: http://localhost:5000
```

Зелёная точка в правом верхнем углу UI = LM Studio подключён и готов.

## Поддерживаемые команды (19 операций)

| Команда (пример) | Операция | OpenCV |
|-----------------|---------|--------|
| «Поверни на 90 градусов» | Поворот без обрезки | `getRotationMatrix2D` |
| «Измени размер до 640x480» | Масштабирование | `resize` (Lanczos4) |
| «Сделай чёрно-белым» | Grayscale | `cvtColor` |
| «Размой сильно» | Gaussian Blur | `GaussianBlur` |
| «Выдели красный канал» | Извлечение RGB-канала | channel split |
| «Инвертируй цвета» | Инверсия | `bitwise_not` |
| «Обнаружь края» | Детектор краёв Canny | `Canny` |
| «Обрежь до 400x400» | Обрезка | array slicing |
| «Увеличь яркость» | Яркость через HSV | `cvtColor` HSV |
| «Зеркально отрази» | Отражение | `flip` |
| «Повысь резкость» | Unsharp-фильтр | `filter2D` |
| «Увеличь контраст» | Масштаб альфа | `convertScaleAbs` |
| «Примени сепию» | Сепия-матрица | `transform` |
| «Пикселизируй» | Пикселизация | double `resize` |
| «Тиснение» | Emboss-фильтр | `filter2D` |
| «Убери шум» | Шумоподавление | `fastNlMeansDenoisingColored` |
| «Выровняй гистограмму» | Выравнивание | `equalizeHist` (YCrCb) |
| «Виньетка» | Виньетка | Gaussian mask |
| «Карандашный эскиз» | Pencil sketch | `divide` + `GaussianBlur` |

## API

### POST /process

```
Content-Type: multipart/form-data

Body:
  image:   <файл изображения>
  command: "текстовая команда на русском или английском"

Response:
{
  "success": true,
  "result_image": "<base64 PNG>",
  "operation": "Поворот на 90°",
  "parsed_command": {"op": "rotate", "angle": 90},
  "original_shape": [480, 640],
  "result_shape": [640, 480]
}
```

### GET /health

Проверка статуса LM Studio. Возвращает список загруженных моделей.

## Смена модели

В `ollama_client.py` — строка `MODEL`:

```python
MODEL = "local-model"  # LM Studio игнорирует имя, используется загруженная в GUI модель
```

Для смены модели достаточно загрузить другую в интерфейсе LM Studio и перезапустить сервер.

## Стек

- **Backend:** Python 3.10+, Flask 3.0
- **CV:** OpenCV 4.10, NumPy
- **LLM:** LM Studio + Meta-Llama-3.1-8B-Instruct-Q4_K_M (GGUF)
- **Frontend:** Vanilla JS, HTML/CSS (без фреймворков)