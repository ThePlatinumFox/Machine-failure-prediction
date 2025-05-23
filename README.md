# Проект: Predictive Maintenance

**Цель:** разработать модель бинарной классификации для предсказания отказов оборудования и оформить её в виде Streamlit-приложения с презентацией.

## Структура
- `app.py` — основной entrypoint для навигации между страницами.
- `analysis_and_model.py` — загрузка, предобработка, обучение и оценка моделей; предсказания.
- `presentation.py` — интерактивная презентация проекта.
- `requirements.txt` — зависимости проекта.
- `data/ai4i2020.csv` — датасет AI4I 2020 для анализа.

## Установка и запуск
```bash
git clone <repo_url>
cd predictive_maintenance_project
pip install -r requirements.txt
streamlit run app.py