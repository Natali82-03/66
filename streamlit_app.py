import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from io import BytesIO
import chardet

# Конфигурация страницы
st.set_page_config(layout="wide", page_title="Демографический анализ")

# Улучшенная функция загрузки данных
@st.cache_data
def load_data(file_name):
    try:
        # Автоопределение кодировки
        with open(file_name, 'rb') as f:
            result = chardet.detect(f.read(10000))
        
        # Пробуем несколько кодировок
        for encoding in [result['encoding'], 'utf-8-sig', 'cp1251']:
            try:
                df = pd.read_csv(file_name, sep=';', encoding=encoding)
                
                # Стандартизация названий столбцов
                df = df.rename(columns=lambda x: x.strip())
                
                # Проверяем возможные варианты названия столбца с именами
                name_col = next((col for col in df.columns 
                               if col.lower() in ['name', 'наименование', 'наименование муниципального образования']), None)
                
                if name_col:
                    df = df.rename(columns={name_col: 'Name'})
                    df['Name'] = df['Name'].str.strip()
                    return df
                    
            except Exception as e:
                continue
                
        st.error(f"Не удалось загрузить файл {file_name}")
        return None
        
    except Exception as e:
        st.error(f"Ошибка при чтении файла {file_name}: {str(e)}")
        return None

# Загрузка всех данных с проверкой
try:
    data_files = {
        "Дети 1-6 лет": "Ch_1_6.csv",
        "Дети 3-18 лет": "Ch_3_18.csv",
        "Дети 5-18 лет": "Ch_5_18.csv",
        "Население 3-79 лет": "Pop_3_79.csv",
        "Среднегодовая численность": "RPop.csv",
        "Инвестиции": "Investment.csv"
    }
    
    data = {}
    for name, file in data_files.items():
        df = load_data(file)
        if df is None or df.empty or 'Name' not in df.columns:
            st.error(f"Проблема с данными: {name}")
            st.stop()
        data[name] = df

except Exception as e:
    st.error(f"Критическая ошибка: {str(e)}")
    st.stop()

# Проверка наличия данных для выбранного показателя
if 'Дети 1-6 лет' not in data or data['Дети 1-6 лет'].empty:
    st.error("Отсутствуют необходимые данные")
    st.stop()

# Сайдбар с настройками
with st.sidebar:
    st.title("Настройки анализа")
    
    # Выбор населенного пункта
    try:
        locations = data['Дети 1-6 лет']['Name'].unique()
        selected_location = st.selectbox(
            "Выберите населенный пункт:",
            locations,
            index=0
        )
    except Exception as e:
        st.error("Не удалось загрузить список населенных пунктов")
        st.stop()
    
    # Выбор показателя
    selected_topic = st.selectbox(
        "Выберите показатель:",
        list(data.keys())[:-1],  # Все кроме инвестиций
        index=0
    )
    
    # Дополнительные опции
    show_forecast = st.checkbox("Показать прогноз", True)
    show_correlation = st.checkbox("Анализ корреляции", True)

# Основной интерфейс
st.title(f"Анализ: {selected_location}")

try:
    # Получаем данные для выбранного показателя
    df = data[selected_topic]
    location_data = df[df['Name'] == selected_location]
    
    if location_data.empty:
        st.warning(f"Нет данных для {selected_location}")
        st.stop()
    
    # Карточки с показателями
    current_year = '2024'
    prev_year = '2023'
    
    if current_year not in df.columns or prev_year not in df.columns:
        st.error("Отсутствуют данные за выбранные годы")
        st.stop()
    
    current_val = location_data[current_year].values[0]
    prev_val = location_data[prev_year].values[0]
    
    cols = st.columns(3)
    cols[0].metric(selected_topic, f"{current_val:,.0f}")
    cols[1].metric("Изменение", f"{current_val - prev_val:+,.0f}")
    cols[2].metric("% изменения", f"{((current_val - prev_val)/prev_val)*100:+.1f}%" if prev_val != 0 else "N/A")

    # График динамики
    years = [str(year) for year in range(2019, 2025) if str(year) in df.columns]
    
    if not years:
        st.warning("Нет данных за указанный период")
    else:
        values = location_data[years].values.flatten()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years,
            y=values,
            name="Фактические данные",
            line=dict(width=3)
        ))
        
        # Прогнозирование
        if show_forecast and len(years) > 1:
            try:
                X = np.array(range(len(years))).reshape(-1, 1)
                model = LinearRegression()
                model.fit(X, values)
                
                future_years = [str(year) for year in range(int(years[-1])+1, int(years[-1])+6)]
                forecast = model.predict(np.array(range(len(years), len(years)+5)).reshape(-1, 1))
                
                fig.add_trace(go.Scatter(
                    x=future_years,
                    y=forecast,
                    name="Прогноз",
                    line=dict(dash='dot')
                ))
            except Exception as e:
                st.error(f"Ошибка прогнозирования: {str(e)}")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Анализ корреляции
    if show_correlation and selected_topic != "Инвестиции":
        st.subheader("Корреляция с инвестициями")
        
        try:
            df_invest = data["Инвестиции"]
            merged_df = pd.merge(
                df, 
                df_invest, 
                on='Name', 
                suffixes=('_demo', '_invest')
            )
            
            if current_year + '_demo' in merged_df.columns and current_year + '_invest' in merged_df.columns:
                fig = px.scatter(
                    merged_df,
                    x=current_year + '_demo',
                    y=current_year + '_invest',
                    trendline="ols",
                    hover_name="Name",
                    title=f"Корреляция {selected_topic} и инвестиций"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                corr = merged_df[current_year + '_demo'].corr(merged_df[current_year + '_invest'])
                st.info(f"Коэффициент корреляции: {corr:.2f}")
            else:
                st.warning("Недостаточно данных для анализа корреляции")
                
        except Exception as e:
            st.error(f"Ошибка анализа корреляции: {str(e)}")

except Exception as e:
    st.error(f"Ошибка обработки данных: {str(e)}")

# Экспорт данных
with st.expander("Экспорт данных"):
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            data[selected_topic].to_excel(writer, sheet_name="Демография")
            data["Инвестиции"].to_excel(writer, sheet_name="Инвестиции")
        
        st.download_button(
            "Скачать данные (Excel)",
            output.getvalue(),
            "демографические_данные.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Ошибка при экспорте данных: {str(e)}")
