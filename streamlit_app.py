import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from io import BytesIO
import chardet
from datetime import datetime

# ======================
# 🎨 КОНФИГУРАЦИЯ СТИЛЕЙ
# ======================
st.set_page_config(
    layout="wide",
    page_title="🔮 Демографический Атлас Орловской области",
    page_icon="📊"
)

# CSS-анимации и стили
st.markdown("""
<style>
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.03); }
    100% { transform: scale(1); }
}
.card {
    animation: fadeIn 1s ease-in;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
.header {
    font-family: 'Arial', sans-serif;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# 📂 ЗАГРУЗКА ДАННЫХ
# ======================
@st.cache_data
def load_data(file_name):
    """Умная загрузка данных с автоопределением кодировки"""
    with open(file_name, 'rb') as f:
        result = chardet.detect(f.read(10000))
    
    encodings = [result['encoding'], 'utf-8-sig', 'cp1251']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_name, sep=';', encoding=encoding)
            # Стандартизация названий
            df = df.rename(columns=lambda x: x.strip())
            if 'Наименование муниципального образования' in df.columns:
                df = df.rename(columns={'Наименование муниципального образования': 'Name'})
            df['Name'] = df['Name'].str.strip()
            return df
        except Exception as e:
            continue
    st.error(f"Не удалось загрузить файл {file_name}")
    return None

# Загрузка всех наборов данных
data_files = {
    "Дети 1-6 лет": "Ch_1_6.csv",
    "Дети 3-18 лет": "Ch_3_18.csv",
    "Дети 5-18 лет": "Ch_5_18.csv",
    "Население 3-79 лет": "Pop_3_79.csv",
    "Среднегодовая численность": "RPop.csv",
    "Инвестиции": "Investment.csv"
}

try:
    with st.spinner('🔍 Загружаем данные...'):
        loaded_data = {}
        for name, file in data_files.items():
            if df := load_data(file):
                loaded_data[name] = df
            else:
                st.error(f"Файл {file} не загружен")
                st.stop()
except Exception as e:
    st.error(f"Критическая ошибка: {str(e)}")
    st.stop()

# ======================
# 🎛️ НАСТРОЙКИ ИНТЕРФЕЙСА
# ======================
with st.sidebar:
    st.title("⚙️ Панель управления")
    st.markdown("---")
    
    # Выбор локации
    selected_location = st.selectbox(
        "📍 Выберите территорию:",
        loaded_data["Дети 1-6 лет"]['Name'].unique(),
        index=0
    )
    
    # Выбор показателя
    selected_topic = st.selectbox(
        "📊 Основной показатель:",
        list(loaded_data.keys())[:-1],  # Все кроме инвестиций
        format_func=lambda x: f"{'👶🧒👦👩🏠'[list(loaded_data.keys()).index(x)]} {x}"
    )
    
    # Дополнительные опции
    st.markdown("---")
    st.markdown("**🔮 Дополнительные анализы:**")
    show_forecast = st.checkbox("Прогноз на 5 лет", True)
    show_correlation = st.checkbox("Корреляция с инвестициями", True)
    show_extremes = st.checkbox("Топ-5 и антирейтинг", True)

# ======================
# 📊 ОСНОВНОЙ ИНТЕРФЕЙС
# ======================
st.markdown(f"""
<div class="header">
    <h1>🔮 Демографический Атлас</h1>
    <h3>📍 {selected_location}</h3>
    <p>{datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
</div>
""", unsafe_allow_html=True)

# 1. КАРТОЧКИ С ПОКАЗАТЕЛЯМИ
current_year = '2024'
prev_year = '2023'
df_topic = loaded_data[selected_topic]

try:
    current_val = df_topic[df_topic['Name'] == selected_location][current_year].values[0]
    prev_val = df_topic[df_topic['Name'] == selected_location][prev_year].values[0]
    delta_val = current_val - prev_val
    delta_pct = (delta_val / prev_val) * 100 if prev_val != 0 else 0
    
    cols = st.columns(3)
    metrics = [
        (f"{current_val:,.0f}", f"{selected_topic} ({current_year})", "#3498db"),
        (f"{delta_val:+,.0f}", "Изменение за год", "#2ecc71" if delta_val >=0 else "#e74c3c"),
        (f"{delta_pct:+.1f}%", "Процент изменения", "#2ecc71" if delta_pct >=0 else "#e74c3c")
    ]
    
    for i, (value, label, color) in enumerate(metrics):
        with cols[i]:
            st.markdown(f"""
            <div class="card" style="border-left: 5px solid {color}">
                <h3 style="margin:0">{label}</h3>
                <h1 style="margin:0">{value}</h1>
            </div>
            """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"Ошибка в данных: {str(e)}")

# 2. ГРАФИК ДИНАМИКИ С ПРОГНОЗОМ
years = [str(year) for year in range(2019, 2025)]
try:
    values = df_topic[df_topic['Name'] == selected_location][years].values.flatten()
    
    fig = go.Figure()
    
    # Основной график
    fig.add_trace(go.Scatter(
        x=years, y=values,
        name="Фактические данные",
        line=dict(width=4),
        marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>"
    ))
    
    # Прогнозирование
    if show_forecast:
        X = np.array(range(len(years))).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, values)
        
        future_years = [str(year) for year in range(2025, 2030)]
        X_future = np.array(range(len(years), len(years)+5)).reshape(-1, 1)
        forecast = model.predict(X_future)
        
        fig.add_trace(go.Scatter(
            x=future_years, y=forecast,
            name="Прогноз",
            line=dict(dash='dot', width=3),
            hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>"
        ))
    
    fig.update_layout(
        title=f"📈 Динамика показателя: {selected_topic}",
        xaxis_title="Год",
        yaxis_title="Значение",
        height=500,
        hovermode="x unified",
        template="plotly_white",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Ошибка построения графика: {str(e)}")

# 3. ТОП-5 И АНТИРЕЙТИНГ
if show_extremes:
    st.subheader("🏆 Топ-5 и антирейтинг")
    
    try:
        last_year = '2024'
        df_topic_sorted = df_topic.sort_values(last_year, ascending=False)
        
        # Топ-5
        top5 = df_topic_sorted.head(5)[['Name', last_year]]
        # Антирейтинг
        bottom5 = df_topic_sorted.tail(5)[['Name', last_year]]
        
        # Создаем графики
        fig_top = px.bar(
            top5, 
            x=last_year, 
            y='Name',
            orientation='h',
            title="Топ-5 территорий",
            labels={last_year: "Значение", "Name": ""},
            color_discrete_sequence=["#27ae60"]
        )
        
        fig_bottom = px.bar(
            bottom5,
            x=last_year,
            y='Name',
            orientation='h',
            title="Антирейтинг 5 территорий",
            labels={last_year: "Значение", "Name": ""},
            color_discrete_sequence=["#e74c3c"]
        )
        
        # Отображаем в колонках
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_top, use_container_width=True)
        with col2:
            st.plotly_chart(fig_bottom, use_container_width=True)
            
    except Exception as e:
        st.error(f"Ошибка при анализе экстремумов: {str(e)}")

# 4. АНАЛИЗ КОРРЕЛЯЦИИ С ИНВЕСТИЦИЯМИ
if show_correlation and selected_topic != "Инвестиции":
    st.subheader("💡 Корреляция с инвестициями")
    
    try:
        df_invest = loaded_data["Инвестиции"]
        merged_df = pd.merge(
            df_topic, df_invest,
            on='Name',
            suffixes=('_demo', '_invest')
        )
        
        last_year = '2024'
        x_col = f"{last_year}_demo"
        y_col = f"{last_year}_invest"
        
        # Проверка данных
        if x_col not in merged_df.columns or y_col not in merged_df.columns:
            st.warning("Недостаточно данных для анализа")
        else:
            # Scatter plot
            fig = px.scatter(
                merged_df,
                x=x_col,
                y=y_col,
                hover_name="Name",
                trendline="ols",
                labels={
                    x_col: f"{selected_topic} ({last_year})",
                    y_col: f"Инвестиции ({last_year})"
                },
                color_discrete_sequence=["#3498db"]
            )
            
            # Выделяем выбранную территорию
            if selected_location in merged_df['Name'].values:
                selected_data = merged_df[merged_df['Name'] == selected_location]
                fig.add_trace(go.Scatter(
                    x=selected_data[x_col],
                    y=selected_data[y_col],
                    name=selected_location,
                    mode='markers',
                    marker=dict(size=12, color='#e74c3c')
                ))
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Расчёт корреляции
            corr = merged_df[x_col].corr(merged_df[y_col])
            
            # Интерпретация
            interpretation = ""
            if corr > 0.7:
                interpretation = "🔹 Сильная прямая связь"
            elif corr > 0.3:
                interpretation = "🔸 Умеренная прямая связь"
            elif corr < -0.7:
                interpretation = "🔺 Сильная обратная связь"
            elif corr < -0.3:
                interpretation = "🔻 Умеренная обратная связь"
            else:
                interpretation = "▫️ Слабая или отсутствует связь"
            
            st.info(f"""
            **Коэффициент корреляции (Пирсон): {corr:.2f}**  
            {interpretation}
            """)
            
    except Exception as e:
        st.error(f"Ошибка анализа корреляции: {str(e)}")

# 5. ЭКСПОРТ ДАННЫХ
with st.expander("💾 Экспорт данных", expanded=False):
    tab1, tab2 = st.tabs(["CSV", "Excel"])
    
    with tab1:
        st.download_button(
            label="📥 Скачать демографические данные (CSV)",
            data=df_topic.to_csv(index=False).encode('utf-8'),
            file_name=f"{selected_topic}_data.csv",
            mime="text/csv"
        )
    
    with tab2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_topic.to_excel(writer, sheet_name=selected_topic[:30])
            loaded_data["Инвестиции"].to_excel(writer, sheet_name="Инвестиции")
        st.download_button(
            label="📊 Скачать полный набор (Excel)",
            data=output.getvalue(),
            file_name="демография_и_инвестиции.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Футер
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d">
    <p>🔮 Аналитический дашборд | Орловская область | {year}</p>
</div>
""".format(year=datetime.now().year), unsafe_allow_html=True)
