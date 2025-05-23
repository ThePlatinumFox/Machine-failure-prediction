import streamlit as st
from streamlit_extras.switch_page_button import switch_page  # or use built-in Page API
import os

# Define navigation
PAGES = {
    "Анализ и модель": "analysis_and_model.py",
    "Презентация": "presentation.py",
}

st.set_page_config(page_title="Predictive Maintenance", layout="wide")

st.sidebar.title("Навигация")
selection = st.sidebar.radio("Перейти на страницу:", list(PAGES.keys()))

page = PAGES[selection]
with open(page, encoding="utf-8") as f:
    code = compile(f.read(), page, 'exec')
    exec(code, globals()) 