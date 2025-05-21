import streamlit as st
from PIL import Image
import numpy as np
from interface import perform_ocr_analysis, check_text_contrast, analyze_clutter

# ========== Текст рекомендаций ==========
good_text = "#### ✅ Ваш интерфейс соответствует рекомендациям UX"

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="UX/UI экспертная система", layout="centered")
st.title("🔍 UX/UI экспертная система")

uploaded_file = st.file_uploader("Загрузите скриншот интерфейса (PNG или JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение")

    # OCR + структура
    st.header("📄 Распознанный текст и контрастность")
    text, ocr_data = perform_ocr_analysis(image)
    st.text_area("Текст с интерфейса", text, height=150)

    img_array = np.array(image)
    contrast_warnings = check_text_contrast(img_array, ocr_data)

    if not contrast_warnings:
        st.success("Текст прошел проверку на контрастность")
    else:
        for w in contrast_warnings:
            st.warning(w)

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

    # Общие выводы
    st.header("Общие рекомендации")

    if contrast_warnings:
        st.markdown("### Рекомендации по контрастности")
        st.markdown("Изучите стандарты WCAG")
        st.markdown("Попробуйте прищурить глаза и попробовать прочитать текст. Если он плохо читается, сделайте его "
                    "контрастнее или увеличьте размер")

    if clutter_info["is_cluttered"]:
        st.markdown("### Рекомендации по загруженности интерфейса")
        st.markdown("#### Определите приоритеты")
        st.markdown("💡 Задай себе вопрос: какую задачу пользователь должен решить на этом экране в первую очередь?")
        st.markdown("❗ Сохрани только ключевые элементы, которые помогают выполнить эту задачу.")
        st.markdown("#### Используйте пустое пространство осознанно")
        st.markdown("🧘 Не бойся пустого пространства — оно помогает глазу ориентироваться.")
        st.markdown("#### Минимизируйте текст")
        st.markdown("✂️ Убирай вводные слова и «воду». Пример: вместо «Чтобы начать, нажмите на эту кнопку» → «Начать».")
        st.markdown("#### Используйте визуальные акценты и цвета с умом")
        st.markdown("🎨 Ограничь палитру до 2–3 основных цветов + нейтральный фон.")
        st.markdown("🚦 Цвет помогает ориентироваться: выдели главное, а второстепенное — приглуши.")
        st.markdown("#### Сократите количество кнопок и действий")
        st.markdown("👇 Объедините похожие действия (например, «Сохранить» и «Готово»).")
        st.markdown("📤 Уберите редко используемые кнопки или переместите их в меню «ещё».")

    if not (contrast_warnings or clutter_info["is_cluttered"]):
        st.success(good_text)
