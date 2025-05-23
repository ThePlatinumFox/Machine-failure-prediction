import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    # Автозагрузка из папки data, если файл существует
    data_path = os.path.join('data', 'ai4i2020.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


def analysis_and_model_page():
    st.title("Анализ данных и модель")

    data = load_data()
    if data is None:
        uploaded = st.file_uploader("Загрузите CSV с данными", type="csv")
        if not uploaded:
            st.info("Пожалуйста, загрузите файл CSV или положите его в data/ai4i2020.csv.")
            return
        data = pd.read_csv(uploaded)

    # Предобработка
    # Убираем недопустимые символы в названиях признаков ([], < и пробелы)
    rename_func = lambda x: (
        x.strip()
        .replace('[','')
        .replace(']','')
        .replace('<','lt')
        .replace(' ','_')
    )
    data.rename(columns=rename_func, inplace=True)
    data = data.drop(columns=['UDI', 'Product_ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], errors='ignore')

    # Целевая переменная
    target_col = 'Machine_failure'

    # Кодирование категориального признака
    # data['Type'] = LabelEncoder().fit_transform(data['Type'])
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    if 'Type' in data.columns:
        data['Type'] = data['Type'].map(type_mapping)
    else:
        st.error("Не удалось найти колонку 'Type' в датасете - проверьте, что она есть после предобработки")
        return

    # Разделение
    X = data.select_dtypes(include=[np.number]).drop(columns=[target_col])
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Масштабирование
    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include='number').columns
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Блок управления обучением — жмём кнопку для запуска и автоматического обновления
    if 'trained' not in st.session_state:
        st.session_state['trained'] = False
    if not st.session_state['trained']:
        if st.button('Обучить модели'):
            st.session_state['trained'] = True
        else:
            return

    # Модели
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'SVM': SVC(kernel='linear', probability=True, random_state=42)
    }
    
    results = {}
    prog = st.progress(0)
    total = len(models)
    for i, (name, model) in enumerate(models.items()):
        with st.spinner(f"Training {name} ({i+1}/{total})..."):
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'conf_matrix': confusion_matrix(y_test, y_pred),
            'report': classification_report(y_test, y_pred, output_dict=False),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'fpr_tpr': roc_curve(y_test, y_proba)
        }
        prog.progress((i+1)/total)

    # Сохраняем в session_state лучшую модель по roc_auc для динамической презентации
    best_name, best_res = max(results.items(), key=lambda kv: kv[1]['roc_auc'])
    st.session_state['best_model'] = best_name
    st.session_state['best_auc'] = best_res['roc_auc']
    st.session_state['best_accuracy'] = best_res['accuracy']
    
    # Отображение
    st.header("Сравнение моделей")

    # Разделение
    for name, res in results.items():
        st.subheader(name)
        st.write(f"Accuracy: {res['accuracy']:.3f}")
        st.write(f"ROC-AUC: {res['roc_auc']:.3f}")
        fig, ax = plt.subplots()
        ax.plot(res['fpr_tpr'][0], res['fpr_tpr'][1], label=f"AUC={res['roc_auc']:.2f}")
        ax.plot([0,1],[0,1],'--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC-кривая: {name}')
        ax.legend()
        st.pyplot(fig)

    # Форма для предсказания
    st.header("Предсказание для новых данных")
    with st.form("predict_form"):
        inputs = {
            'Type': st.selectbox('Type', [0,1,2]),  # L,M,H -> 0,1,2
            'Air temperature [K]': st.number_input('Air temperature [K]'),
            'Process temperature [K]': st.number_input('Process temperature [K]'),
            'Rotational speed [rpm]': st.number_input('Rotational speed [rpm]'),
            'Torque [Nm]': st.number_input('Torque [Nm]'),
            'Tool wear [min]': st.number_input('Tool wear [min]'),
        }
        submitted = st.form_submit_button('Предсказать')
        if submitted:
            df_input = pd.DataFrame([inputs])
            df_input.rename(columns=rename_func, inplace=True)
            df_input[num_cols] = scaler.transform(df_input[num_cols])
            pred = models['Random Forest'].predict(df_input)
            proba = models['Random Forest'].predict_proba(df_input)[:,1]
            st.write(f"Отказ: {int(pred[0])}")
            st.write(f"Вероятность: {proba[0]:.2f}")

if __name__ == "__main__":
    analysis_and_model_page()