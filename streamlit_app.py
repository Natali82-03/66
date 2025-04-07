import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# Конфигурация страницы
st.set_page_config(layout="wide", page_title="Демография и инвестиции")

# --- Загрузка данных ---
@st.cache_data
def load_data(file_name):
    with open(file_name, 'rb') as f:
        result = chardet.detect(f.read(10000))
    try:
        df = pd.read_csv(file_name, sep=';', encoding=result['encoding'])
    except UnicodeDecodeError:
        df = pd.read_csv(file_name, sep=';', encoding='cp1251')
    
    df = df.rename(columns=lambda x: x.strip())
    if 'Наименование муниципального образования' in df.columns:
        df = df.rename(columns={'Наименование муниципального образования': 'Name'})
    df['Name'] = df['Name'].str.strip()
    return df

# Загрузка всех файлов
try:
    ch_1_6 = load_data('Ch_1_6.csv')
    ch_3_18 = load_data('Ch_3_18.csv')
    ch_5_18 = load_data('Ch_5_18.csv')
    pop_3_79 = load_data('Pop_3_79.csv')
    rpop = load_data('RPop.csv')
    investments = load_data('Investment.csv')  # Новый файл с инвестициями
except Exception as e:
    st.error(f"Ошибка загрузки данных: {str(e)}")
    st.stop()

# Словарь данных
data_dict = {
    "Дети 1-6 лет": (ch_1_6, "#1f77b4"),
    "Дети 3-18 лет": (ch_3_18, "#ff7f0e"),
    "Дети 5-18 лет": (ch_5_18, "#2ca02c"),
    "Население 3-79 лет": (pop_3_79, "#d62728"),
    "Среднегодовая численность": (rpop, "#9467bd"),
    "Инвестиции": (investments, "#17becf")  # Добавляем инвестиции
}

# --- Боковая панель ---
with st.sidebar:
    st.title("Настройки анализа")
    selected_location = st.selectbox("Населённый пункт:", ch_1_6['Name'].unique())
    selected_topic = st.selectbox("Категория населения:", list(data_dict.keys())[:-1])
    show_forecast = st.checkbox("Показать прогноз на 5 лет", True)
    show_correlation = st.checkbox("Анализ корреляции с инвестициями", True)

# --- Основной интерфейс ---
st.title(f"📊 {selected_location}: демография и инвестиции")

# 1. Индикаторы динамики
current_year = '2024'
prev_year = '2023'
df_topic = data_dict[selected_topic][0]
current_val = df_topic[df_topic['Name'] == selected_location][current_year].values[0]
prev_val = df_topic[df_topic['Name'] == selected_location][prev_year].values[0]
delta_pct = ((current_val - prev_val) / prev_val) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(f"{selected_topic} ({current_year})", f"{current_val:,.0f} чел.")
with col2:
    st.metric("Изменение за год", f"{current_val - prev_val:+,.0f} чел.")
with col3:
    st.metric("Процент изменения", f"{delta_pct:+.1f}%")

# 2. График динамики с прогнозом
years = [str(year) for year in range(2019, 2025)]
values = df_topic[df_topic['Name'] == selected_location][years].values.flatten()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=years, y=values, 
    name="Исторические данные",
    line=dict(width=4)
))

# Прогнозирование
if show_forecast:
    X = np.array(range(len(years))).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, values)
    
    future_years = [str(year) for year in range(2025, 2029)]
    X_future = np.array(range(len(years), len(years)+4)).reshape(-1, 1)
    forecast = model.predict(X_future)
    
    fig.add_trace(go.Scatter(
        x=future_years, y=forecast,
        name="Прогноз",
        line=dict(dash='dot', width=3)
    ))

fig.update_layout(
    title=f"Динамика: {selected_topic}",
    height=500,
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# 3. Анализ корреляции с инвестициями
if show_correlation:
    st.subheader("📈 Корреляция с инвестициями")
    
    # Собираем данные для scatter plot
    df_invest = data_dict["Инвестиции"][0]
    merged_df = pd.merge(
        df_topic, df_invest, 
        on='Name', 
        suffixes=('_pop', '_invest')
    )
    
    # Выбираем последний доступный год
    last_year = '2024'
    fig_scatter = px.scatter(
        merged_df, 
        x=f"{last_year}_pop", 
        y=f"{last_year}_invest",
        hover_data=['Name'],
        trendline="ols",
        labels={
            f"{last_year}_pop": f"{selected_topic}, чел.",
            f"{last_year}_invest": "Инвестиции, руб."
        }
    )
    
    # Добавляем выбранный пункт
    selected_point = merged_df[merged_df['Name'] == selected_location]
    fig_scatter.add_trace(go.Scatter(
        x=selected_point[f"{last_year}_pop"],
        y=selected_point[f"{last_year}_invest"],
        name=selected_location,
        marker=dict(size=12, color='red')
    ))
    
    fig_scatter.update_layout(height=600)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Расчёт коэффициента корреляции
    corr = merged_df[f"{last_year}_pop"].corr(merged_df[f"{last_year}_invest"])
    st.info(f"Коэффициент корреляции между {selected_topic.lower()} и инвестициями: **{corr:.2f}**")

# 4. Экспорт данных
with st.expander("📤 Экспорт данных"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_topic.to_excel(writer, sheet_name="Демография")
        investments.to_excel(writer, sheet_name="Инвестиции")
    st.download_button(
        "Скачать все данные (Excel)",
        output.getvalue(),
        "демография_и_инвестиции.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
