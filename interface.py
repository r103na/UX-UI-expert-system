import pytesseract
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import cv2
def preprocess_image_for_ocr(img):
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    new_image = Image.fromarray(gray)
    return new_image

def perform_ocr_analysis(image: Image.Image):
    custom_config = r'--oem 3 --psm 4'

    new_image = preprocess_image_for_ocr(image)
    text = pytesseract.image_to_string(new_image, config=custom_config, lang="eng+rus")
    data = pytesseract.image_to_data(new_image, config=custom_config, output_type=pytesseract.Output.DICT)
    return text, data

def relative_luminance(rgb):
    def channel_lum(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * channel_lum(r) + 0.7152 * channel_lum(g) + 0.0722 * channel_lum(b)

def contrast_ratio(lum1, lum2):
    L1, L2 = max(lum1, lum2), min(lum1, lum2)
    return (L1 + 0.05) / (L2 + 0.05)

def check_text_contrast(img_array: np.ndarray, ocr_data):
    warnings = []
    for i in range(len(ocr_data["text"])):
        try:
            conf = int(ocr_data["conf"][i])
        except ValueError:
            continue

        text = ocr_data["text"][i].strip()

        if len(text) <= 1:
            continue

        if conf > 50 and text:
            x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
            region = img_array[y:y + h, x:x + w]

            if region.size == 0:
                continue

            if region.shape[2] == 4:  # remove alpha
                region = region[:, :, :3]

            pixels = region.reshape(-1, 3)

            try:
                kmeans = KMeans(n_clusters=2, n_init="auto", random_state=0).fit(pixels)
                cluster_centers = kmeans.cluster_centers_
                labels = kmeans.labels_

                counts = np.bincount(labels)
                text_rgb = cluster_centers[np.argmin(counts)]   # меньшая группа — предположительно текст
                bg_rgb = cluster_centers[np.argmax(counts)]   # большая группа — фон

                text_lum = relative_luminance(text_rgb)
                bg_lum = relative_luminance(bg_rgb)

                contrast = contrast_ratio(text_lum, bg_lum)

                if contrast < 4.5:
                    warnings.append(
                        f"⚠️ Низкий контраст текста '{text}': {contrast:.2f} (менее 4.5)"
                    )
            except Exception as e:
                print(f"KMeans error: {e}")
                continue

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
