"""
image_processor.py — выполняет OpenCV операции на основе структурированной команды.
"""
import cv2
import numpy as np
from typing import Tuple


def apply_command(image: np.ndarray, command: dict) -> Tuple[np.ndarray, str]:
    """
    Принимает изображение (numpy array) и команду (dict).
    Возвращает (обработанное изображение, описание выполненной операции).
    """
    op = command.get("op", "unknown")

    if op == "rotate":
        return _rotate(image, command)

    elif op == "resize":
        return _resize(image, command)

    elif op == "grayscale":
        return _grayscale(image)

    elif op == "blur":
        return _blur(image, command)

    elif op == "channel":
        return _extract_channel(image, command)

    elif op == "invert":
        return _invert(image)

    elif op == "edges":
        return _edges(image, command)

    elif op == "crop":
        return _crop(image, command)

    elif op == "brightness":
        return _brightness(image, command)

    elif op == "flip":
        return _flip(image, command)

    elif op == "sharpen":
        return _sharpen(image)

    elif op == "contrast":
        return _contrast(image, command)

    elif op == "sepia":
        return _sepia(image)

    elif op == "pixelate":
        return _pixelate(image, command)

    elif op == "emboss":
        return _emboss(image)

    elif op == "denoise":
        return _denoise(image)

    elif op == "histogram":
        return _equalize_histogram(image)

    elif op == "vignette":
        return _vignette(image, command)

    elif op == "sketch":
        return _sketch(image)

    elif op == "unknown":
        msg = command.get("message", "Неизвестная команда")
        raise ValueError(f"Команда не распознана: {msg}")

    else:
        raise ValueError(f"Операция '{op}' не реализована")


# ─────────────────────── Реализации операций ───────────────────────

def _rotate(img: np.ndarray, cmd: dict) -> Tuple[np.ndarray, str]:
    angle = float(cmd.get("angle", 90))
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    # Вычисляем новые размеры чтобы изображение не обрезалось
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2
    result = cv2.warpAffine(img, matrix, (new_w, new_h))
    return result, f"Поворот на {angle}°"


def _resize(img: np.ndarray, cmd: dict) -> Tuple[np.ndarray, str]:
    w = int(cmd.get("width", img.shape[1]))
    h = int(cmd.get("height", img.shape[0]))
    result = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return result, f"Масштаб до {w}×{h}px"


def _grayscale(img: np.ndarray) -> Tuple[np.ndarray, str]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # обратно в BGR для единообразия
    return result, "Перевод в оттенки серого"


def _blur(img: np.ndarray, cmd: dict) -> Tuple[np.ndarray, str]:
    intensity = int(cmd.get("intensity", 5))
    # Ядро должно быть нечётным
    kernel = max(1, intensity * 2 - 1)
    result = cv2.GaussianBlur(img, (kernel, kernel), 0)
    return result, f"Размытие (интенсивность {intensity})"


def _extract_channel(img: np.ndarray, cmd: dict) -> Tuple[np.ndarray, str]:
    color = cmd.get("color", "red").lower()
    channel_map = {"blue": 0, "green": 1, "red": 2}
    idx = channel_map.get(color, 2)
    result = np.zeros_like(img)
    result[:, :, idx] = img[:, :, idx]
    color_ru = {"red": "красный", "green": "зелёный", "blue": "синий"}.get(color, color)
    return result, f"Выделение {color_ru} канала"


def _invert(img: np.ndarray) -> Tuple[np.ndarray, str]:
    result = cv2.bitwise_not(img)
    return result, "Инверсия цветов"


def _edges(img: np.ndarray, cmd: dict) -> Tuple[np.ndarray, str]:
    t1 = int(cmd.get("threshold1", 50))
    t2 = int(cmd.get("threshold2", 150))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, t1, t2)
    result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return result, f"Обнаружение краёв Canny ({t1}, {t2})"


def _crop(img: np.ndarray, cmd: dict) -> Tuple[np.ndarray, str]:
    h, w = img.shape[:2]
    x = int(cmd.get("x", 0))
    y = int(cmd.get("y", 0))
    cw = int(cmd.get("width", w // 2))
    ch = int(cmd.get("height", h // 2))
    # Защита от выхода за границы
    x2 = min(x + cw, w)
    y2 = min(y + ch, h)
    result = img[y:y2, x:x2]
    return result, f"Обрезка: ({x},{y}) {cw}×{ch}px"


def _brightness(img: np.ndarray, cmd: dict) -> Tuple[np.ndarray, str]:
    value = int(cmd.get("value", 50))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value) if value > 0 else cv2.subtract(v, abs(value))
    v = np.clip(v, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    sign = "+" if value >= 0 else ""
    return result, f"Яркость {sign}{value}"


def _flip(img: np.ndarray, cmd: dict) -> Tuple[np.ndarray, str]:
    direction = cmd.get("direction", "horizontal").lower()
    flip_code = 1 if direction == "horizontal" else 0
    result = cv2.flip(img, flip_code)
    dir_ru = "горизонтальное" if direction == "horizontal" else "вертикальное"
    return result, f"Зеркальное отражение ({dir_ru})"


def _sharpen(img: np.ndarray) -> Tuple[np.ndarray, str]:
    kernel = np.array([
        [0, -1,  0],
        [-1,  5, -1],
        [0, -1,  0]
    ], dtype=np.float32)
    result = cv2.filter2D(img, -1, kernel)
    return result, "Повышение резкости"


def _contrast(img: np.ndarray, cmd: dict) -> Tuple[np.ndarray, str]:
    value = float(cmd.get("value", 1.5))
    result = cv2.convertScaleAbs(img, alpha=value, beta=0)
    return result, f"Контраст ×{value}"


def _sepia(img: np.ndarray) -> Tuple[np.ndarray, str]:
    kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])
    result = cv2.transform(img, kernel)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result, "Сепия-фильтр"


def _pixelate(img: np.ndarray, cmd: dict) -> Tuple[np.ndarray, str]:
    h, w = img.shape[:2]
    size = int(cmd.get("size", 15))
    size = max(2, size)
    small = cv2.resize(img, (w // size, h // size), interpolation=cv2.INTER_LINEAR)
    result = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return result, f"Пикселизация (блок {size}px)"


def _emboss(img: np.ndarray) -> Tuple[np.ndarray, str]:
    kernel = np.array([
        [-2, -1,  0],
        [-1,  1,  1],
        [ 0,  1,  2]
    ], dtype=np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    embossed = cv2.filter2D(gray, -1, kernel) + 128
    embossed = np.clip(embossed, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)
    return result, "Тиснение (Emboss)"


def _denoise(img: np.ndarray) -> Tuple[np.ndarray, str]:
    result = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return result, "Шумоподавление"


def _equalize_histogram(img: np.ndarray) -> Tuple[np.ndarray, str]:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return result, "Выравнивание гистограммы"


def _vignette(img: np.ndarray, cmd: dict) -> Tuple[np.ndarray, str]:
    h, w = img.shape[:2]
    strength = float(cmd.get("strength", 1.5))
    X = cv2.getGaussianKernel(w, w / strength)
    Y = cv2.getGaussianKernel(h, h / strength)
    mask = Y * X.T
    mask = mask / mask.max()
    result = img.copy().astype(np.float32)
    for i in range(3):
        result[:, :, i] *= mask
    return result.astype(np.uint8), "Виньетка"


def _sketch(img: np.ndarray) -> Tuple[np.ndarray, str]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, cv2.bitwise_not(blurred), scale=256.0)
    result = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    return result, "Карандашный эскиз"

