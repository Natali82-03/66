import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from io import BytesIO
import chardet
from datetime import datetime

# Конфигурация страницы
st.set_page_config(layout="wide", page_title="Демографический анализ")

# Загрузка данных
@st.cache_data
def load_data(file_name):
    with open(file_name, 'rb') as f:
        result = chardet.detect(f.read(10000))
    try:
        return pd.read_csv(file_name, sep=';', encoding=result['encoding'])
    except:
        try:
            return pd.read_csv(file_name, sep=';', encoding='utf-8-sig')
        except:
            return pd.read_csv(file_name, sep=';', encoding='cp1251')

try:
    data = {
        "Дети 1-6 лет": load_data('Ch_1_6.csv'),
        "Дети 3-18 лет": load_data('Ch_3_18.csv'),
        "Дети 5-18 лет": load_data('Ch_5_18.csv'),
        "Население 3-79 лет": load_data('Pop_3_79.csv'),
        "Среднегодовая численность": load_data('RPop.csv'),
        "Инвестиции": load_data('Investment.csv')
    }
    
    # Проверка загрузки данных
    for name, df in data.items():
        if df is None or df.empty:
            st.error(f"Ошибка загрузки данных: {name}")
            st.stop()
except Exception as e:
    st.error(f"Критическая ошибка: {str(e)}")
    st.stop()

# Сайдбар
with st.sidebar:
    st.title("Настройки")
    selected_location = st.selectbox("Населенный пункт", data["Дети 1-6 лет"]['Name'].unique())
    selected_topic = st.selectbox("Показатель", list(data.keys())[:-1])
    show_forecast = st.checkbox("Показать прогноз", True)
    show_correlation = st.checkbox("Анализ корреляции", True)

# Основной интерфейс
st.title(f"Демографический анализ: {selected_location}")

try:
    # Карточки с метриками
    current_year = '2024'
    prev_year = '2023'
    df = data[selected_topic]
    current_val = df[df['Name'] == selected_location][current_year].values[0]
    prev_val = df[df['Name'] == selected_location][prev_year].values[0]
    
    cols = st.columns(3)
    cols[0].metric(selected_topic, f"{current_val:,.0f}")
    cols[1].metric("Изменение", f"{current_val - prev_val:+,.0f}")
    cols[2].metric("% изменения", f"{((current_val - prev_val)/prev_val)*100:+.1f}%")

    # График динамики
    years = [str(year) for year in range(2019, 2025)]
    values = df[df['Name'] == selected_location][years].values.flatten()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=values, name="Факт"))
    
    if show_forecast:
        model = LinearRegression()
        model.fit(np.array(range(len(years))).reshape(-1,1), values)
        future = model.predict(np.array(range(len(years), len(years)+5)).reshape(-1,1))
        fig.add_trace(go.Scatter(
            x=[str(y) for y in range(2025, 2030)],
            y=future,
            name="Прогноз",
            line=dict(dash='dot')
        ))
    
    st.plotly_chart(fig, use_container_width=True)

    # Анализ корреляции
    if show_correlation and selected_topic != "Инвестиции":
        st.subheader("Корреляция с инвестициями")
        merged = pd.merge(
            df, 
            data["Инвестиции"], 
            on='Name', 
            suffixes=('_pop', '_inv')
        )
        
        fig = px.scatter(
            merged, 
            x=f"{current_year}_pop", 
            y=f"{current_year}_inv",
            trendline="ols",
            hover_name="Name"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        corr = merged[f"{current_year}_pop"].corr(merged[f"{current_year}_inv"])
        st.info(f"Коэффициент корреляции: {corr:.2f}")

except Exception as e:
    st.error(f"Ошибка обработки данных: {str(e)}")

# Экспорт данных
with st.expander("Экспорт"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        data[selected_topic].to_excel(writer, sheet_name="Демография")
        data["Инвестиции"].to_excel(writer, sheet_name="Инвестиции")
    st.download_button(
        "Скачать Excel",
        output.getvalue(),
        "данные.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
