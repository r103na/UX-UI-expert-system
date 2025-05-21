import streamlit as st
from PIL import Image
import numpy as np
import pytesseract
from sklearn.cluster import KMeans
import cv2

# ========== Текст рекомендаций ==========
def preprocess_image_for_ocr(img):
    img_array = np.array(img)

    # Увеличение контраста
    alpha = 2.5
    beta = 0
    contrast_img = cv2.convertScaleAbs(img_array, alpha=alpha, beta=beta)

    gray = cv2.cvtColor(contrast_img, cv2.COLOR_BGR2GRAY)

    new_image = Image.fromarray(gray)
    return new_image


def perform_ocr_analysis(image: Image.Image):
    custom_config = r'--oem 3 --psm 4'
    min_text_length = 1

    new_image = preprocess_image_for_ocr(image)
    data = pytesseract.image_to_data(new_image, config=custom_config, output_type=pytesseract.Output.DICT)

    # Фильтрация: удаляем все записи с текстом длиной 1 или меньше
    filtered_data = {key: [] for key in data.keys()}

    for i in range(len(data["text"])):
        if len(data["text"][i].strip()) > min_text_length:
            for key in data.keys():
                filtered_data[key].append(data[key][i])

    return filtered_data


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
        confidence_rate = 60

        try:
            conf = int(ocr_data["conf"][i])
        except ValueError:
            continue

        text = ocr_data["text"][i].strip()

        if conf >= confidence_rate and text:
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
                text_rgb = cluster_centers[np.argmin(counts)]  # меньшая группа — предположительно текст
                bg_rgb = cluster_centers[np.argmax(counts)]  # большая группа — фон

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

    if num_elements > 25:
        clutter_score += 1
    elif num_elements > 50:
        clutter_score += 2

    if area_ratio > 0.05:
        clutter_score += 1

    if text_density > 0.00025:
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


# ========== Текст рекомендаций ==========
good_text = "#### ✅ Ваш интерфейс соответствует рекомендациям UX"

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="UX/UI экспертная система", layout="centered")
st.title("🔍 UX/UI экспертная система")

uploaded_file = st.file_uploader("Загрузите скриншот интерфейса (PNG или JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение")

    st.divider()

    # OCR + структура
    st.header("📄 Распознанный текст и контрастность")
    ocr_data = perform_ocr_analysis(image)

    img_array = np.array(image)
    contrast_warnings = check_text_contrast(img_array, ocr_data)

    if not contrast_warnings:
        st.success("✅ Текст прошел проверку на контрастность")
    else:
        for w in contrast_warnings:
            st.warning(w)

    st.divider()

    # Перегрузка интерфейса
    clutter_info = analyze_clutter(ocr_data, image.size)
    st.subheader("📊 Анализ перегруженности интерфейса")
    st.write(f"Элементов текста: {clutter_info['num_elements']}")
    st.write(f"Площадь текста к общей: {clutter_info['area_ratio']}")
    st.write(f"Плотность текста: {clutter_info['text_density']}")
    st.write(f"Кластеризация: {clutter_info['cluster_score']}")
    if clutter_info["is_cluttered"]:
        st.warning("⚠ **Интерфейс перегружен**")
    else:
        st.success("✅ Интерфейс не перегружен")

    st.divider()

    # Общие выводы
    st.header("Общие рекомендации")

    if contrast_warnings:
        with st.chat_message("user"):
            st.markdown("### Рекомендации по контрастности")
            st.markdown("Изучите стандарты WCAG и используйте плагины в Figma для проверки")
            st.markdown("Попробуйте прищурить глаза и попробовать прочитать текст. Если он плохо читается, сделайте его "
                        "контрастнее или увеличьте размер")

    if clutter_info["is_cluttered"]:
        with st.chat_message("user"):
            st.markdown("### Рекомендации по загруженности интерфейса")
            st.markdown("#### Определите приоритеты")
            st.markdown("💡 Задай себе вопрос: какую задачу пользователь должен решить на этом экране в первую очередь?")
            st.markdown("❗ Сохрани только ключевые элементы, которые помогают выполнить эту задачу.")
            st.markdown("#### Используйте пустое пространство осознанно")
            st.markdown("🧘 Не бойся пустого пространства — оно помогает глазу ориентироваться.")
            st.markdown("#### Минимизируйте текст")
            st.markdown("✂️ Убирай вводные слова и «воду». Пример: вместо «Чтобы начать, нажмите на эту кнопку» → "
                        "«Начать».")
            st.markdown("#### Используйте визуальные акценты и цвета с умом")
            st.markdown("🎨 Ограничь палитру до 2–3 основных цветов + нейтральный фон.")
            st.markdown("🚦 Цвет помогает ориентироваться: выдели главное, а второстепенное — приглуши.")
            st.markdown("#### Сократите количество кнопок и действий")
            st.markdown("👇 Объедините похожие действия (например, «Сохранить» и «Готово»).")
            st.markdown("📤 Уберите редко используемые кнопки или переместите их в меню «ещё».")

    if not (contrast_warnings or clutter_info["is_cluttered"]):
        st.success(good_text)
