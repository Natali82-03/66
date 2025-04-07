import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from io import BytesIO
import chardet  # Добавляем недостающий импорт

# Конфигурация страницы
st.set_page_config(layout="wide", page_title="Демография и инвестиции")

# --- Улучшенная загрузка данных ---
@st.cache_data
def load_data(file_name):
    # Автоматическое определение кодировки
    with open(file_name, 'rb') as f:
        rawdata = f.read()
        result = chardet.detect(rawdata)
    
    try:
        df = pd.read_csv(file_name, sep=';', encoding=result['encoding'])
    except:
        # Попробуем другие распространенные кодировки, если автоматическое определение не сработало
        try:
            df = pd.read_csv(file_name, sep=';', encoding='utf-8-sig')
        except:
            df = pd.read_csv(file_name, sep=';', encoding='cp1251')
    
    # Стандартизация названий столбцов
    df = df.rename(columns=lambda x: x.strip())
    if 'Наименование муниципального образования' in df.columns:
        df = df.rename(columns={'Наименование муниципального образования': 'Name'})
    df['Name'] = df['Name'].str.strip()
    return df

# Загрузка всех файлов с обработкой ошибок
try:
    with st.spinner('Загрузка данных...'):
        ch_1_6 = load_data('Ch_1_6.csv')
        ch_3_18 = load_data('Ch_3_18.csv')
        ch_5_18 = load_data('Ch_5_18.csv')
        pop_3_79 = load_data('Pop_3_79.csv')
        rpop = load_data('RPop.csv')
        investments = load_data('Investment.csv')
except Exception as e:
    st.error(f"""Ошибка загрузки данных: {str(e)}
            Проверьте:
            1. Все файлы находятся в той же папке
            2. Названия файлов совпадают:
               - Ch_1_6.csv
               - Ch_3_18.csv
               - Ch-5-18.csv
               - Pop_3_79.csv
               - RPop.csv
               - Investment.csv""")
    st.stop()

# Словарь данных с человеко-читаемыми названиями
data_dict = {
    "Дети 1-6 лет": (ch_1_6, "#1f77b4", "👶"),
    "Дети 3-18 лет": (ch_3_18, "#ff7f0e", "🧒"),
    "Дети 5-18 лет": (ch_5_18, "#2ca02c", "👦"),
    "Население 3-79 лет": (pop_3_79, "#d62728", "👩"),
    "Среднегодовая численность": (rpop, "#9467bd", "🏠"),
    "Инвестиции": (investments, "#17becf", "💰")  # Добавляем инвестиции
}

# --- Боковая панель ---
with st.sidebar:
    st.title("⚙️ Настройки анализа")
    selected_location = st.selectbox(
        "Выберите населённый пункт:",
        ch_1_6['Name'].unique(),
        index=0
    )
    
    selected_topic = st.selectbox(
        "Выберите демографический показатель:",
        list(data_dict.keys())[:-1],  # Все кроме инвестиций
        format_func=lambda x: f"{data_dict[x][2]} {x}"  # Добавляем иконки
    )
    
    analysis_options = st.multiselect(
        "Дополнительные анализы:",
        ["Прогноз на 5 лет", "Корреляция с инвестициями"],
        default=["Прогноз на 5 лет", "Корреляция с инвестициями"]
    )

# --- Основной интерфейс ---
st.title(f"{data_dict[selected_topic][2]} Демография и инвестиции: {selected_location}")

# 1. Индикаторы динамики в карточках
current_year = '2024'
prev_year = '2023'
df_topic, color, icon = data_dict[selected_topic]

try:
    current_val = df_topic[df_topic['Name'] == selected_location][current_year].values[0]
    prev_val = df_topic[df_topic['Name'] == selected_location][prev_year].values[0]
    delta_val = current_val - prev_val
    delta_pct = (delta_val / prev_val) * 100 if prev_val != 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label=f"{icon} {selected_topic} ({current_year})",
            value=f"{current_val:,.0f}",
            delta_color="normal"
        )
    with col2:
        st.metric(
            label="📈 Изменение за год",
            value=f"{delta_val:+,.0f}",
            delta_color="inverse" if delta_val < 0 else "normal"
        )
    with col3:
        st.metric(
            label="📊 Процент изменения",
            value=f"{delta_pct:+.1f}%",
            delta_color="inverse" if delta_pct < 0 else "normal"
        )
except Exception as e:
    st.warning(f"Не удалось загрузить данные для выбранного населённого пункта: {str(e)}")

# 2. Интерактивный график с прогнозом
years = [str(year) for year in range(2019, 2025)]
values = df_topic[df_topic['Name'] == selected_location][years].values.flatten()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=years, y=values,
    name="Исторические данные",
    line=dict(color=color, width=4),
    hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>"
))

# Прогнозирование если выбрано
if "Прогноз на 5 лет" in analysis_options:
    try:
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
        
        # Добавляем доверительный интервал
        residuals = values - model.predict(X)
        stdev = np.std(residuals)
        fig.add_trace(go.Scatter(
            x=future_years,
            y=forecast + 1.96*stdev,
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=future_years,
            y=forecast - 1.96*stdev,
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            name="95% доверительный интервал",
            opacity=0.2
        ))
    except Exception as e:
        st.error(f"Ошибка прогнозирования: {str(e)}")

fig.update_layout(
    title=f"{icon} Динамика: {selected_topic}",
    xaxis_title="Год",
    yaxis_title="Численность" if selected_topic != "Инвестиции" else "Рублей",
    height=500,
    hovermode="x unified",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# 3. Анализ корреляции с инвестициями
if "Корреляция с инвестициями" in analysis_options and selected_topic != "Инвестиции":
    st.subheader("📈 Корреляция между демографией и инвестициями")
    
    df_invest = data_dict["Инвестиции"][0]
    merged_df = pd.merge(
        df_topic, df_invest,
        on='Name',
        suffixes=('_demo', '_invest')
    )
    
    # Выбираем последний доступный год
    last_year = '2024'
    
    # Создаем масштабированные данные для сравнения
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(merged_df[[f"{last_year}_demo", f"{last_year}_invest"]])
    merged_df["scaled_demo"] = scaled_data[:, 0]
    merged_df["scaled_invest"] = scaled_data[:, 1]
    
    # График корреляции
    fig_corr = px.scatter(
        merged_df,
        x=f"{last_year}_demo",
        y=f"{last_year}_invest",
        trendline="ols",
        hover_name="Name",
        labels={
            f"{last_year}_demo": f"{selected_topic} ({last_year})",
            f"{last_year}_invest": f"Инвестиции ({last_year})"
        },
        color_discrete_sequence=[color]
    )
    
    # Выделяем выбранный пункт
    selected_data = merged_df[merged_df['Name'] == selected_location]
    fig_corr.add_trace(go.Scatter(
        x=selected_data[f"{last_year}_demo"],
        y=selected_data[f"{last_year}_invest"],
        name=selected_location,
        mode='markers',
        marker=dict(size=12, color='red')
    ))
    
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # График сравнения нормированных значений
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Bar(
        x=merged_df["Name"],
        y=merged_df["scaled_demo"],
        name=f"{selected_topic} (нормировано)",
        marker_color=color
    ))
    fig_comparison.add_trace(go.Bar(
        x=merged_df["Name"],
        y=merged_df["scaled_invest"],
        name="Инвестиции (нормировано)",
        marker_color="#17becf"
    ))
    fig_comparison.update_layout(
        title="Сравнение нормированных показателей",
        barmode='group',
        height=600,
        xaxis_title="Населённый пункт",
        yaxis_title="Нормированное значение",
        xaxis={'categoryorder':'total descending'}
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Расчёт статистики
    corr_coef = merged_df[f"{last_year}_demo"].corr(merged_df[f"{last_year}_invest"])
    st.info(f"""
        **Коэффициент корреляции Пирсона**: {corr_coef:.2f}
        - 1.0: Полная прямая корреляция
        - 0.0: Нет корреляции
        - -1.0: Полная обратная корреляция
    """)

# 4. Экспорт данных
with st.expander("💾 Экспорт данных", expanded=False):
    tab1, tab2 = st.tabs(["CSV", "Excel"])
    
    with tab1:
        st.download_button(
            label="Скачать демографические данные (CSV)",
            data=df_topic.to_csv(index=False).encode('utf-8'),
            file_name=f"{selected_topic.replace(' ', '_')}.csv",
            mime="text/csv"
        )
        
        st.download_button(
            label="Скачать инвестиционные данные (CSV)",
            data=investments.to_csv(index=False).encode('utf-8'),
            file_name="investment_data.csv",
            mime="text/csv"
        )
    
    with tab2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_topic.to_excel(writer, sheet_name=selected_topic[:30])
            investments.to_excel(writer, sheet_name="Инвестиции")
        st.download_button(
            label="Скачать все данные (Excel)",
            data=output.getvalue(),
            file_name="демография_и_инвестиции.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
