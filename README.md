# 👁 Vision AI — Image Processing

> Система обработки изображений через естественный язык.  
> Вы пишете команду по-русски — ИИ понимает, что нужно сделать, и выполняет это через OpenCV.

---

## Что делает проект

Пользователь загружает изображение, вводит команду на естественном языке — например _«сделай карандашный эскиз»_ или _«выдели красный канал»_ — и получает обработанный результат. Никаких кнопок "повернуть на 90°", никаких ползунков. Просто слова.

Под капотом локальная языковая модель (LLM) парсит команду и превращает её в структурированную JSON-инструкцию, которую выполняет OpenCV.

---

## Как это работает

```
Пользователь вводит команду
        ↓
   Flask (app.py)
   принимает изображение + текст
        ↓
command_parser.py
   отправляет промпт в LM Studio
        ↓
   LLM (Llama 3.1 8B Instruct)
   возвращает JSON-команду
   например: {"op": "sketch"}
        ↓
image_processor.py
   выполняет операцию через OpenCV
        ↓
   Результат — base64 PNG
        ↓
   Web UI отображает результат
   показывает JSON, время, размеры
```

---

## Архитектура проекта

```
vision-ai-image-processor/
├── app.py                # Flask-сервер: маршруты, валидация, логирование
├── .gitignore            # Для игнора не нужных файлов чтобы не забивать мусором гитхаб репозиторий
├── command_parser.py     # Prompt engineering: текст → JSON через LLM
├── image_processor.py    # 19 OpenCV-операций
├── ollama_client.py      # HTTP-клиент к LM Studio API
├── static/
│   └── index.html        # Web UI: светлая тема, drag-and-drop, сравнение
└── requirements.txt
```

### Связка фронтенда и бэкенда

Фронтенд (`index.html`) — чистый HTML/CSS/JS без фреймворков. Работает прямо в браузере.

- Пользователь загружает фото через drag-and-drop или клик
- Вводит команду в текстовое поле (или нажимает быструю кнопку)
- JS отправляет `POST /process` с `multipart/form-data` (изображение + команда)
- Flask принимает запрос, отдаёт на обработку LLM → OpenCV
- Возвращает JSON с base64-изображением, описанием операции, временем выполнения
- UI отображает результат, позволяет сравнить с оригиналом и скачать

### API

```
POST /process
  image:   <файл>
  command: "текстовая команда"

→ {
    "success": true,
    "result_image": "<base64 PNG>",
    "operation": "Карандашный эскиз",
    "parsed_command": {"op": "sketch"},
    "original_shape": [1080, 1920],
    "result_shape": [1080, 1920]
  }

GET /health
→ статус LM Studio + список загруженных моделей
```

---

## Поддерживаемые операции (19)

| Команда | Операция | Метод OpenCV |
|--------|---------|-------------|
| «Поверни на 90°» | Поворот без обрезки | `getRotationMatrix2D` |
| «Измени размер до 640x480» | Масштабирование | `resize` Lanczos4 |
| «Сделай чёрно-белым» | Grayscale | `cvtColor` |
| «Размой» | Gaussian Blur | `GaussianBlur` |
| «Выдели красный канал» | RGB-канал | channel split |
| «Инвертируй цвета» | Инверсия | `bitwise_not` |
| «Обнаружь края» | Canny | `Canny` |
| «Обрежь» | Crop | array slicing |
| «Увеличь яркость» | Яркость | HSV `cvtColor` |
| «Отрази горизонтально» | Flip | `flip` |
| «Повысь резкость» | Sharpen | `filter2D` |
| «Увеличь контраст» | Contrast | `convertScaleAbs` |
| «Сепия» | Sepia | `transform` |
| «Пикселизируй» | Pixelate | double `resize` |
| «Тиснение» | Emboss | `filter2D` |
| «Убери шум» | Denoise | `fastNlMeansDenoisingColored` |
| «Выровняй гистограмму» | Histogram EQ | `equalizeHist` YCrCb |
| «Виньетка» | Vignette | Gaussian mask |
| «Карандашный эскиз» | Sketch | `divide` + blur |

---

## Стек технологий

| Слой | Технология |
|------|-----------|
| Backend | Python 3.14, Flask 3.0 |
| Computer Vision | OpenCV 4.10, NumPy |
| LLM | LM Studio + Meta-Llama-3.1-8B-Instruct Q4_K_M (GGUF) |
| Frontend | Vanilla JS, HTML5, CSS3 (без фреймворков) |
| Туннель | Cloudflare Tunnel |

---

## Установка и запуск , если заинтригованы сами на своём ноуте сделать всё

### 1. LM Studio

1. Скачать: [lmstudio.ai](https://lmstudio.ai) - если его у вас нет 
2. Найти и скачать модель: `Meta-Llama-3.1-8B-Instruct-Q4_K_M` (~4.92 GB) - хорошая и умная модель 
3. Вкладка **Local Server** → выбрать модель → **Start Server** , после выбираете эту модель и запускаете её **Load Model** и      выбираете скачанный **Meta-Llama-3.1-8B-Instruct-Q4_K_M**
4. Сервер запустится на `http://localhost:1234`

### 2. Python

```bash
pip install -r requirements.txt
python app.py
```

Открыть в браузере: `http://localhost:5000`

---

## Удалённый доступ

Проект работает локально, но с помощью **Cloudflare Tunnel** можно получить публичную ссылку — и открыть UI на любом устройстве в мире без каких-либо настроек со стороны пользователя.

### Как получить ссылку

Убедитесь что Flask запущен (`python app.py`), затем в отдельном терминале:

```bash
C:\cloudflared\cloudflared.exe tunnel --url http://localhost:5000
```

Через несколько секунд в терминале появится строка:

```
https://abc-something-random.trycloudflare.com
```

Отправьте эту ссылку — получатель открывает её в браузере и получает полноценный доступ к интерфейсу. Все вычисления при этом происходят на вашем компьютере: LM Studio обрабатывает команды, OpenCV обрабатывает изображения.

```
Пользователь (любая точка мира)
        ↓
  браузер → https://xxx.trycloudflare.com
        ↓
  Cloudflare Tunnel
        ↓
  localhost:5000 (ваш Flask)
        ↓
  LM Studio + OpenCV (ваш ноутбук)
```

> Ссылка активна пока запущены `cloudflared` и `python app.py`.  
> При каждом новом запуске `cloudflared` ссылка меняется.  
> Бесплатно, регистрация не требуется.