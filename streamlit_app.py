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
# 🎨 КОНФИГУРАЦИЯ СТРАНИЦЫ
# ======================
st.set_page_config(
    layout="wide",
    page_title="🔮 Демографический Атлас",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# CSS стили и анимации
st.markdown("""
<style>
    .header {
        font-family: 'Arial', sans-serif;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        background: white;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .positive {
        border-left: 5px solid #2ecc71;
    }
    .negative {
        border-left: 5px solid #e74c3c;
    }
    .neutral {
        border-left: 5px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# 📂 ФУНКЦИИ ЗАГРУЗКИ ДАННЫХ
# ======================
@st.cache_data
def safe_load_data(file_name):
    """Безопасная загрузка данных с обработкой ошибок"""
    try:
        # Определение кодировки
        with open(file_name, 'rb') as f:
            rawdata = f.read(10000)
            result = chardet.detect(rawdata)
        
        # Попробуем разные кодировки
        encodings = [result['encoding'], 'utf-8-sig', 'cp1251']
        for encoding in encodings:
            try:
                df = pd.read_csv(file_name, sep=';', encoding=encoding)
                if not df.empty:
                    # Стандартизация столбцов
                    df = df.rename(columns=lambda x: x.strip())
                    if 'Наименование муниципального образования' in df.columns:
                        df = df.rename(columns={'Наименование муниципального образования': 'Name'})
                    df['Name'] = df['Name'].str.strip()
                    return df
            except:
                continue
        return None
    except Exception as e:
        st.error(f"Критическая ошибка при чтении файла {file_name}: {str(e)}")
        return None

def load_all_datasets():
    """Загрузка всех необходимых наборов данных"""
    data_files = {
        "Дети 1-6 лет": "Ch_1_6.csv",
        "Дети 3-18 лет": "Ch_3_18.csv",
        "Дети 5-18 лет": "Ch_5_18.csv",
        "Население 3-79 лет": "Pop_3_79.csv",
        "Среднегодовая численность": "RPop.csv",
        "Инвестиции": "Investment.csv"
    }
    
    loaded_data = {}
    for name, file in data_files.items():
        with st.spinner(f'Загрузка {name}...'):
            df = safe_load_data(file)
            if df is not None and not df.empty:
                loaded_data[name] = df
            else:
                st.error(f"Не удалось загрузить {file}")
                return None
    return loaded_data

# ======================
# 📊 ФУНКЦИИ ВИЗУАЛИЗАЦИИ
# ======================
def create_metric_card(value, label, delta=None):
    """Создает красивую карточку с метрикой"""
    delta_class = "positive" if (delta and delta >= 0) else "negative" if delta else "neutral"
    st.markdown(f"""
    <div class="metric-card {delta_class}">
        <h3 style="margin:0; color: #7f8c8d">{label}</h3>
        <h1 style="margin:0">{value}</h1>
        {f'<p style="margin:0; color: {"#2ecc71" if delta >=0 else "#e74c3c"}">{delta:+,.0f}</p>' if delta is not None else ''}
    </div>
    """, unsafe_allow_html=True)

def plot_timeseries(years, values, topic, color, forecast_years=None, forecast_values=None):
    """Построение графика временного ряда с прогнозом"""
    fig = go.Figure()
    
    # Основной график
    fig.add_trace(go.Scatter(
        x=years, y=values,
        name="Фактические данные",
        line=dict(color=color, width=4),
        mode='lines+markers',
        hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>"
    ))
    
    # Прогноз если есть
    if forecast_years and forecast_values:
        fig.add_trace(go.Scatter(
            x=forecast_years, y=forecast_values,
            name="Прогноз",
            line=dict(color=color, width=3, dash='dot'),
            hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>"
        ))
    
    fig.update_layout(
        title=f"📈 Динамика: {topic}",
        xaxis_title="Год",
        yaxis_title="Значение",
        height=500,
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

# ======================
# 📌 ОСНОВНОЙ КОД
# ======================
def main():
    # Загрузка данных
    data = load_all_datasets()
    if not data:
        st.error("Не удалось загрузить необходимые данные. Проверьте файлы.")
        return
    
    # ======================
    # 🎛️ ПАНЕЛЬ УПРАВЛЕНИЯ
    # ======================
    with st.sidebar:
        st.title("⚙️ Панель управления")
        st.markdown("---")
        
        # Выбор локации
        locations = data["Дети 1-6 лет"]['Name'].unique()
        selected_location = st.selectbox(
            "📍 Выберите территорию:",
            locations,
            index=0
        )
        
        # Выбор показателя
        topics = list(data.keys())[:-1]  # Все кроме инвестиций
        selected_topic = st.selectbox(
            "📊 Основной показатель:",
            topics,
            index=0
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
        <p>Обновлено: {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. КАРТОЧКИ С ПОКАЗАТЕЛЯМИ
    current_year = '2024'
    prev_year = '2023'
    df_topic = data[selected_topic]
    
    try:
        # Получаем данные для выбранной локации
        location_data = df_topic[df_topic['Name'] == selected_location]
        if location_data.empty:
            st.warning(f"Нет данных для {selected_location}")
        else:
            current_val = location_data[current_year].values[0]
            prev_val = location_data[prev_year].values[0]
            delta_val = current_val - prev_val
            
            # Отображаем карточки
            cols = st.columns(3)
            with cols[0]:
                create_metric_card(f"{current_val:,.0f}", f"{selected_topic} ({current_year})")
            with cols[1]:
                create_metric_card(f"{delta_val:+,.0f}", "Изменение за год", delta_val)
            with cols[2]:
                delta_pct = (delta_val / prev_val) * 100 if prev_val != 0 else 0
                create_metric_card(f"{delta_pct:+.1f}%", "Процент изменения", delta_pct)
            
            # 2. ГРАФИК ДИНАМИКИ С ПРОГНОЗОМ
            years = [str(year) for year in range(2019, 2025)]
            values = location_data[years].values.flatten()
            
            # Прогнозирование
            forecast_years, forecast_values = None, None
            if show_forecast and len(years) == len(values):
                try:
                    X = np.array(range(len(years))).reshape(-1, 1)
                    model = LinearRegression()
                    model.fit(X, values)
                    
                    forecast_years = [str(year) for year in range(2025, 2030)]
                    X_future = np.array(range(len(years), len(years)+5)).reshape(-1, 1)
                    forecast_values = model.predict(X_future)
                except Exception as e:
                    st.error(f"Ошибка прогнозирования: {str(e)}")
            
            # Построение графика
            fig = plot_timeseries(
                years, values, 
                selected_topic, "#3498db",
                forecast_years, forecast_values
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 3. ТОП-5 И АНТИРЕЙТИНГ
            if show_extremes:
                st.subheader("🏆 Топ-5 и антирейтинг")
                
                try:
                    last_year_data = df_topic[['Name', current_year]].dropna()
                    top5 = last_year_data.nlargest(5, current_year)
                    bottom5 = last_year_data.nsmallest(5, current_year)
                    
                    # Создаем графики
                    fig1 = px.bar(
                        top5, x=current_year, y='Name',
                        orientation='h',
                        title=f"Топ-5 по {selected_topic}",
                        color_discrete_sequence=["#27ae60"]
                    )
                    
                    fig2 = px.bar(
                        bottom5, x=current_year, y='Name',
                        orientation='h',
                        title="Антирейтинг 5 территорий",
                        color_discrete_sequence=["#e74c3c"]
                    )
                    
                    # Отображаем в колонках
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig2, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Ошибка при анализе экстремумов: {str(e)}")
            
            # 4. АНАЛИЗ КОРРЕЛЯЦИИ С ИНВЕСТИЦИЯМИ
            if show_correlation and selected_topic != "Инвестиции":
                st.subheader("💡 Корреляция с инвестициями")
                
                try:
                    df_invest = data["Инвестиции"]
                    merged_df = pd.merge(
                        df_topic, df_invest,
                        on='Name',
                        suffixes=('_demo', '_invest')
                    )
                    if not merged_df.empty:
                        last_year = current_year
                        x_col = f"{last_year}_demo"
                        y_col = f"{last_year}_invest"
                        
                        if x_col in merged_df.columns and y_col in merged_df.columns:
                            # Scatter plot
                            fig = px.scatter(
                                merged_df,
                                x=x_col,
                                y=y_col,
                                trendline="ols",
                                hover_name="Name",
                                labels={
                                    x_col: f"{selected_topic}",
                                    y_col: "Инвестиции"
                                }
                            )
                            
                            # Выделяем выбранную локацию
                            if selected_location in merged_df['Name'].values:
                                selected_point = merged_df[merged_df['Name'] == selected_location]
                                fig.add_trace(go.Scatter(
                                    x=selected_point[x_col],
                                    y=selected_point[y_col],
                                    mode='markers',
                                    marker=dict(size=12, color='red'),
                                    name=selected_location
                                ))
                            
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Расчет корреляции
                            corr = merged_df[x_col].corr(merged_df[y_col])
                            st.info(f"""
                            **Коэффициент корреляции**: {corr:.2f}
                            - От 0.7 до 1.0: Сильная прямая связь
                            - От 0.3 до 0.7: Умеренная связь
                            - От -0.3 до 0.3: Слабая или отсутствует связь
                            - От -1.0 до -0.7: Сильная обратная связь
                            """)
                        else:
                            st.warning("Отсутствуют необходимые столбцы данных")
                    else:
                        st.warning("Нет данных для анализа корреляции")
                except Exception as e:
                    st.error(f"Ошибка анализа корреляции: {str(e)}")
    
    except Exception as e:
        st.error(f"Критическая ошибка: {str(e)}")
    
    # 5. ЭКСПОРТ ДАННЫХ
    with st.expander("💾 Экспорт данных", expanded=False):
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_topic.to_excel(writer, sheet_name=selected_topic[:30])
                data["Инвестиции"].to_excel(writer, sheet_name="Инвестиции")
            
            st.download_button(
                label="📥 Скачать все данные (Excel)",
                data=output.getvalue(),
                file_name="демография_и_инвестиции.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Ошибка при экспорте данных: {str(e)}")

if __name__ == "__main__":
    main()
