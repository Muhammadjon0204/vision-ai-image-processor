"""
app.py — Flask сервер, точка входа приложения.
Маршруты:
  GET  /          → UI (index.html)
  POST /process   → принимает изображение + команду, возвращает результат
  GET  /health    → статус Ollama
"""
import os
import base64
import json
import logging
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

from command_parser import parse_command
from image_processor import apply_command

# ─── Настройка ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB максимум

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def decode_image(file_data: bytes) -> np.ndarray:
    """Декодирует байты в numpy array (BGR, как требует OpenCV)."""
    nparr = np.frombuffer(file_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось декодировать изображение")
    return img


def encode_image_base64(img: np.ndarray) -> str:
    """Кодирует numpy array в base64 PNG."""
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise ValueError("Не удалось закодировать изображение")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    """Проверка доступности LM Studio."""
    import requests as req
    try:
        r = req.get("http://localhost:1234/v1/models", timeout=3)
        models = [m["id"] for m in r.json().get("data", [])]
        return jsonify({"status": "ok", "lmstudio": "running", "models": models})
    except Exception as e:
        return jsonify({"status": "error", "lmstudio": "not running", "error": str(e)}), 503


@app.route("/process", methods=["POST"])
def process_image():
    """
    Основной эндпоинт.
    Принимает: multipart/form-data с полями 'image' и 'command'
    Возвращает: JSON {
        success: bool,
        result_image: base64 PNG,
        operation: str,
        parsed_command: dict,
        original_shape: [h, w],
        result_shape: [h, w]
    }
    """
    # Валидация входных данных
    if "image" not in request.files:
        return jsonify({"success": False, "error": "Файл изображения не передан"}), 400

    file = request.files["image"]
    command_text = request.form.get("command", "").strip()

    if not command_text:
        return jsonify({"success": False, "error": "Команда не указана"}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": f"Неподдерживаемый формат файла"}), 400

    try:
        # 1. Декодируем изображение
        image_bytes = file.read()
        image = decode_image(image_bytes)
        original_shape = list(image.shape[:2])  # [h, w]
        logger.info(f"Входное изображение: {original_shape}, команда: '{command_text}'")

        # 2. LLM парсит команду
        parsed_cmd = parse_command(command_text)
        logger.info(f"Распознанная команда: {parsed_cmd}")

        # 3. OpenCV выполняет операцию
        result_image, operation_description = apply_command(image, parsed_cmd)
        result_shape = list(result_image.shape[:2])  # [h, w]
        logger.info(f"Операция выполнена: {operation_description}, результат: {result_shape}")

        # 4. Кодируем результат
        result_b64 = encode_image_base64(result_image)

        return jsonify({
            "success": True,
            "result_image": result_b64,
            "operation": operation_description,
            "parsed_command": parsed_cmd,
            "original_shape": original_shape,
            "result_shape": result_shape
        })

    except ValueError as e:
        logger.warning(f"Ошибка обработки: {e}")
        return jsonify({"success": False, "error": str(e)}), 422

    except RuntimeError as e:
        logger.error(f"Ошибка Ollama: {e}")
        return jsonify({"success": False, "error": str(e)}), 503

    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Внутренняя ошибка: {e}"}), 500


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Image AI Controller запущен")
    logger.info("UI: http://localhost:5000")
    logger.info("Убедитесь что Ollama запущен: ollama serve")
    logger.info("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
