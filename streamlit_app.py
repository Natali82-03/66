import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import chardet
import base64

# Конфигурация страницы
st.set_page_config(layout="wide", page_title="Демография и инвестиции")

# Функция для установки фона
def set_background_with_overlay(image_file, overlay_opacity=0.7):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{b64_encoded});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .main .block-container {{
            background-color: rgba(255, 255, 255, {overlay_opacity});
            border-radius: 10px;
            padding: 2rem;
            margin-top: 2rem;
            margin-bottom: 2rem;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# Функция для добавления логотипа в угол
def add_corner_logo():
    with open("logo.png", "rb") as f:
        logo_data = base64.b64encode(f.read()).decode()
    
    st.markdown(
        f"""
        <style>
            .corner-logo {{
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 100;
                opacity: 0.8;
                transition: opacity 0.3s;
            }}
            .corner-logo:hover {{
                opacity: 1;
            }}
        </style>
        <div class="corner-logo">
            <img src="data:image/png;base64,{logo_data}" width="80">
        </div>
        """,
        unsafe_allow_html=True
    )

# Установка фона и логотипа
set_background_with_overlay("fon.jpg", overlay_opacity=0.85)
add_corner_logo()

# --- Улучшенная загрузка данных ---
@st.cache_data
def load_data(file_name):
    with open(file_name, 'rb') as f:
        result = chardet.detect(f.read(10000))
    try:
        df = pd.read_csv(file_name, sep=';', encoding=result['encoding'])
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_name, sep=';', encoding='utf-8-sig')
        except:
            df = pd.read_csv(file_name, sep=';', encoding='cp1251')
    
    # Стандартизация данных
    df = df.rename(columns=lambda x: x.strip())
    if 'Наименование муниципального образования' in df.columns:
        df = df.rename(columns={'Наименование муниципального образования': 'Name'})
    df['Name'] = df['Name'].str.strip()
    return df

# Загрузка всех файлов с обработкой ошибок
try:
    with st.spinner('Загрузка данных...'):
        data_sources = {
            "Дети 1-6 лет": 'Ch_1_6.csv',
            "Дети 3-18 лет": 'Ch_3_18.csv', 
            "Дети 5-18 лет": 'Ch_5_18.csv',
            "Население 3-79 лет": 'Pop_3_79.csv',
            "Среднегодовая численность": 'RPop.csv',
            "Инвестиции": 'Investment.csv'
        }
        
        loaded_data = {}
        for name, file in data_sources.items():
            loaded_data[name] = load_data(file)
            
        ch_1_6 = loaded_data["Дети 1-6 лет"]
        ch_3_18 = loaded_data["Дети 3-18 лет"]
        ch_5_18 = loaded_data["Дети 5-18 лет"]
        pop_3_79 = loaded_data["Население 3-79 лет"]
        rpop = loaded_data["Среднегодовая численность"]
        investments = loaded_data["Инвестиции"]

except Exception as e:
    st.error(f"""Ошибка загрузки данных: {str(e)}
            Проверьте:
            1. Все файлы находятся в той же папке
            2. Названия файлов:
               - Ch_1_6.csv
               - Ch_3_18.csv
               - Ch-5-18.csv
               - Pop_3_79.csv
               - RPop.csv
               - Investment.csv""")
    st.stop()

# Словарь данных
data_dict = {
    "Дети 1-6 лет": (ch_1_6, "#1f77b4", "👶"),
    "Дети 3-18 лет": (ch_3_18, "#ff7f0e", "🧒"), 
    "Дети 5-18 лет": (ch_5_18, "#2ca02c", "👦"),
    "Население 3-79 лет": (pop_3_79, "#d62728", "👩"),
    "Среднегодовая численность": (rpop, "#9467bd", "🏠"),
    "Инвестиции": (investments, "#17becf", "💰")
}

# --- Боковая панель ---
with st.sidebar:
    # Создаем две колонки для логотипов
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Логотип Минцифры
        st.image("min.png", width=200)
    
    with col2:
        # Логотип университета
        st.image("ogu.png", width=200)
    
    st.title("⚙️ Настройки анализа")
    
    # Выбор локации
    selected_location = st.selectbox(
        "Выберите населённый пункт:",
        ch_1_6['Name'].unique(),
        index=0
    )
    
    # Выбор показателя
    selected_topic = st.selectbox(
        "Выберите показатель:",
        list(data_dict.keys())[:-1],  # Все кроме инвестиций
        format_func=lambda x: f"{data_dict[x][2]} {x}"
    )
    
    # Дополнительные опции
    analysis_options = st.multiselect(
        "Дополнительные анализы:",
        ["Прогноз на 5 лет", "Корреляция с инвестициями"],
        default=["Прогноз на 5 лет", "Корреляция с инвестициями"]
    )

# --- Основной интерфейс ---
st.title(f"{data_dict[selected_topic][2]} Анализ: {selected_location}")

# Инициализация cols перед блоком try
cols = st.columns(3)

# 1. Карточки с показателями
current_year = '2024'
prev_year = '2023'

try:
    df_topic, color, icon = data_dict[selected_topic]
    
    # Проверка наличия выбранного года в данных
    if current_year not in df_topic.columns or prev_year not in df_topic.columns:
        raise ValueError(f"Данные за {current_year} или {prev_year} год отсутствуют")
    
    current_val = df_topic[df_topic['Name'] == selected_location][current_year].values[0]
    prev_val = df_topic[df_topic['Name'] == selected_location][prev_year].values[0]
    delta_val = current_val - prev_val
    delta_pct = (delta_val / prev_val) * 100 if prev_val != 0 else 0
    
    with cols[0]:
        st.metric(f"{icon} {selected_topic} ({current_year})", 
                 f"{current_val:,.0f}")
    with cols[1]:
        st.metric("Изменение за год", 
                 f"{delta_val:+,.0f}",
                 delta_color="inverse" if delta_val < 0 else "normal")
    with cols[2]:
        st.metric("Процент изменения",
                 f"{delta_pct:+.1f}%",
                 delta_color="inverse" if delta_pct < 0 else "normal")

except Exception as e:
    st.error(f"Ошибка при загрузке показателей: {str(e)}")
    # Заполняем карточки заглушками в случае ошибки
    with cols[0]:
        st.metric(f"{icon} {selected_topic} ({current_year})", "N/A")
    with cols[1]:
        st.metric("Изменение за год", "N/A")
    with cols[2]:
        st.metric("Процент изменения", "N/A")

# 2. График динамики с прогнозом
try:
    years = [str(year) for year in range(2019, 2025)]
    values = df_topic[df_topic['Name'] == selected_location][years].values.flatten()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=values,
        name="Фактические данные",
        line=dict(color=color, width=4),
        hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>"
    ))

    # Прогнозирование
    if "Прогноз на 5 лет" in analysis_options:
        X = np.array(range(len(years))).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, values)
        
        future_years = [str(year) for year in range(2025, 2030)]
        X_future = np.array(range(len(years), len(years)+5)).reshape(-1, 1)
        forecast = model.predict(X_future)
        
        fig.add_trace(go.Scatter(
            x=future_years, y=forecast,
            name="Прогноз",
            line=dict(dash='dot', color=color, width=3),
            hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>"
        ))
    
    fig.update_layout(
        title=f"{icon} Динамика показателя",
        xaxis_title="Год",
        yaxis_title="Значение",
        height=500,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Ошибка при построении графика: {str(e)}")

# 3. Анализ корреляции с инвестициями
if "Корреляция с инвестициями" in analysis_options and selected_topic != "Инвестиции":
    st.subheader("📈 Корреляция с инвестициями")
    
    try:
        df_invest = data_dict["Инвестиции"][0]
        merged_df = pd.merge(
            df_topic, df_invest,
            on='Name',
            suffixes=('_demo', '_invest')
        )
        
        last_year = '2024'
        
        # Проверка наличия данных
        if f"{last_year}_demo" not in merged_df.columns or f"{last_year}_invest" not in merged_df.columns:
            raise ValueError("Отсутствуют необходимые колонки для анализа корреляции")
        
        # Основной график корреляции
        fig_corr = px.scatter(
            merged_df,
            x=f"{last_year}_demo",
            y=f"{last_year}_invest",
            trendline="ols",
            hover_name="Name",
            labels={
                f"{last_year}_demo": f"{selected_topic}",
                f"{last_year}_invest": "Инвестиции"
            },
            color_discrete_sequence=[color]
        )
        
        # Выделяем выбранный пункт
        if selected_location in merged_df['Name'].values:
            selected_data = merged_df[merged_df['Name'] == selected_location]
            fig_corr.add_trace(go.Scatter(
                x=selected_data[f"{last_year}_demo"],
                y=selected_data[f"{last_year}_invest"],
                name=selected_location,
                mode='markers',
                marker=dict(size=12, color='red')
            ))
        
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Расчёт корреляции
        corr_coef = merged_df[f"{last_year}_demo"].corr(
            merged_df[f"{last_year}_invest"]
        )
        
        st.info(f"""
        **Коэффициент корреляции**: {corr_coef:.2f}
        - От 0.7 до 1.0: Сильная прямая связь
        - От 0.3 до 0.7: Умеренная связь
        - От -0.3 до 0.3: Слабая связь
        - От -1.0 до -0.7: Сильная обратная связь
        """)
        
    except Exception as e:
        st.error(f"Ошибка при анализе корреляции: {str(e)}")

# 4. Экспорт данных
with st.expander("💾 Экспорт данных"):
    tab1, tab2 = st.tabs(["CSV", "Excel"])
    
    with tab1:
        st.download_button(
            "Скачать демографические данные (CSV)",
            df_topic.to_csv(index=False).encode('utf-8'),
            f"{selected_topic}.csv",
            "text/csv"
        )
    
    with tab2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_topic.to_excel(writer, sheet_name=selected_topic[:30])
            investments.to_excel(writer, sheet_name="Инвестиции")
        st.download_button(
            "Скачать все данные (Excel)",
            output.getvalue(),
            "демография_и_инвестиции.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
