"""
command_parser.py — преобразует естественный язык в структурированную команду
через локальную LLM (Ollama).
"""
import json
import re
from ollama_client import ask_ollama

SYSTEM_PROMPT = """You are a computer vision command parser. 
The user will describe an image operation in natural language (Russian or English).
Your task is to extract the operation and return ONLY a valid JSON object.

Available operations and their JSON format:
1. rotate: {"op": "rotate", "angle": <degrees>}
2. resize: {"op": "resize", "width": <px>, "height": <px>}
3. grayscale: {"op": "grayscale"}
4. blur: {"op": "blur", "intensity": <1-20>}
5. channel: {"op": "channel", "color": "<red|green|blue>"}
6. invert: {"op": "invert"}
7. edges: {"op": "edges", "threshold1": <50>, "threshold2": <150>}
8. crop: {"op": "crop", "x": <px>, "y": <px>, "width": <px>, "height": <px>}
9. brightness: {"op": "brightness", "value": <-100 to 100>}
10. flip: {"op": "flip", "direction": "<horizontal|vertical>"}
11. sharpen: {"op": "sharpen"}
12. contrast: {"op": "contrast", "value": <0.5 to 3.0>}
13. sepia: {"op": "sepia"}
14. pixelate: {"op": "pixelate", "size": <5-50>}
15. emboss: {"op": "emboss"}
16. denoise: {"op": "denoise"}
17. histogram: {"op": "histogram"}
18. vignette: {"op": "vignette", "strength": <1.0-3.0>}
19. sketch: {"op": "sketch"}

Rules:
- Return ONLY the JSON object, no explanation, no markdown, no extra text
- If unsure about a parameter, use sensible defaults
- "поверни на 90" → {"op": "rotate", "angle": 90}
- "сделай серым" → {"op": "grayscale"}
- "выдели красный канал" → {"op": "channel", "color": "red"}
- "сепия" or "старое фото" → {"op": "sepia"}
- "пикселизируй" → {"op": "pixelate", "size": 15}
- "карандашный рисунок" or "эскиз" → {"op": "sketch"}
- "виньетка" → {"op": "vignette", "strength": 1.5}
- "убери шум" or "шумоподавление" → {"op": "denoise"}
- "тиснение" or "emboss" → {"op": "emboss"}
- "выровняй гистограмму" → {"op": "histogram"}
- If command is unclear, return: {"op": "unknown", "message": "unclear command"}

User command:"""


def parse_command(user_input: str) -> dict:
    """
    Принимает текстовую команду пользователя,
    возвращает структурированный dict с операцией.
    """
    full_prompt = SYSTEM_PROMPT + f"\n{user_input}"
    raw_response = ask_ollama(full_prompt)

    # Извлекаем JSON из ответа (LLM иногда добавляет лишний текст)
    json_match = re.search(r'\{[^{}]+\}', raw_response, re.DOTALL)
    if not json_match:
        return {"op": "unknown", "message": f"Не удалось разобрать ответ LLM: {raw_response}"}

    try:
        command = json.loads(json_match.group())
        return command
    except json.JSONDecodeError:
        return {"op": "unknown", "message": f"Некорректный JSON от LLM: {raw_response}"}
