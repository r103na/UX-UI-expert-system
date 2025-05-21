import pytesseract
from PIL import Image
import numpy as np

def perform_ocr_analysis(image: Image.Image):
    text = pytesseract.image_to_string(image, lang="eng+rus")
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    return text, data

def check_text_contrast(img_array: np.ndarray, ocr_data):
    warnings = []
    for i in range(len(ocr_data["text"])):
        try:
            conf = int(ocr_data["conf"][i])
        except ValueError:
            continue

        if conf > 50 and ocr_data["text"][i].strip():
            x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
            region = img_array[y:y + h, x:x + w]
            brightness = np.mean(region)

            if brightness < 60 or brightness > 195:
                warnings.append(
                    f"⚠️ Низкий контраст текста '{ocr_data['text'][i]}'. Яркость: {brightness:.1f}"
                )
    return warnings

def analyze_clutter(ocr_data, image_size):
    width, height = image_size
    image_area = width * height
    num_elements = 0
    total_text_area = 0
    boxes = []

    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i]
        if not text.strip():
            continue
        w = ocr_data['width'][i]
        h = ocr_data['height'][i]
        area = w * h
        total_text_area += area
        num_elements += 1
        boxes.append((ocr_data['left'][i], ocr_data['top'][i], w, h))

    text_density = num_elements / image_area
    area_ratio = total_text_area / image_area

    # Коэффициент кластеризации: расстояния между блоками
    def avg_distance(boxes):
        if len(boxes) < 2:
            return 0
        distances = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                x1, y1, _, _ = boxes[i]
                x2, y2, _, _ = boxes[j]
                dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                distances.append(dist)
        return sum(distances) / len(distances)

    cluster_score = avg_distance(boxes) / (image_size[0] + image_size[1])

    # Критерии перегруженности
    clutter_score = 0
    if num_elements > 20:
        clutter_score += 1
    if num_elements > 50:
        clutter_score += 2
    if area_ratio > 0.15:
        clutter_score += 1
    if text_density > 0.000025:
        clutter_score += 1
    if cluster_score < 0.1:
        clutter_score += 1

    is_cluttered = clutter_score >= 2

    return {
        "num_elements": num_elements,
        "area_ratio": round(area_ratio, 3),
        "text_density": round(text_density, 6),
        "cluster_score": round(cluster_score, 3),
        "is_cluttered": is_cluttered
    }
