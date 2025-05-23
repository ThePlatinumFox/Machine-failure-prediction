import streamlit as st
import reveal_slides as rs

# st.set_page_config(page_title="Презентация проекта", layout="wide")

st.title("Презентация: Прогнозирование отказов оборудования")

presentation_md = f"""
# Прогнозирование отказов оборудования
---
## Введение
- Цель: предсказать отказ оборудования (Machine failure).
- Датасет: AI4I 2020, 10k записей (data/ai4i2020.csv).
---
## Этапы работы
1. Загрузка и предобработка данных
2. Обучение моделей
3. Оценка и сравнение
4. Веб-приложение на Streamlit
---
## Результаты
- **Лучшая модель**: {st.session_state.get('best_model',    '—')}
- **ROC-AUC**:         {st.session_state.get('best_auc',     0):.2f}
- **Accuracy**:        {st.session_state.get('best_accuracy',0):.2f}
---
## Заключение
- Возможные улучшения: подбор гиперпараметров, глубокое обучение.
"""

with st.sidebar:
    st.header("Параметры слайдов")
    theme = st.selectbox("Тема", ["black","white","league","beige","night"])
    transition = st.selectbox("Переход", ["slide","convex","concave","zoom","none"])

rs.slides(
    presentation_md,
    theme=theme,
    config={"transition": transition},
    markdown_props={"data-separator-vertical": "^---$"}
)