import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from io import BytesIO
import chardet

# Конфигурация страницы
st.set_page_config(layout="wide", page_title="Демография и инвестиции")

# ... (остальные функции и загрузка данных остаются без изменений)

# 3. Анализ корреляции с инвестициями
if "Корреляция с инвестициями" in analysis_options and selected_topic != "Инвестиции":
    st.subheader("📈 Корреляция с инвестициями")
    
    try:
        df_invest = data_dict["Инвестиции"][0]
        merged_df = pd.merge(
            df_topic, df_invest,
            on='Name',
            suffixes=('_demo', '_invest'))
        
        last_year = '2024'
        
        # Создаем базовый scatter plot
        fig_corr = px.scatter(
            merged_df,
            x=f"{last_year}_demo",
            y=f"{last_year}_invest",
            hover_name="Name",
            labels={
                f"{last_year}_demo": f"{selected_topic}",
                f"{last_year}_invest": "Инвестиции"
            },
            color_discrete_sequence=[color]
        )
        
        # Ручной расчет линии тренда
        x = merged_df[f"{last_year}_demo"].values
        y = merged_df[f"{last_year}_invest"].values
        if len(x) > 1:  # Проверка на достаточное количество точек
            coefficients = np.polyfit(x, y, 1)
            trendline = np.poly1d(coefficients)
            x_trend = np.linspace(x.min(), x.max(), 100)
            
            fig_corr.add_trace(go.Scatter(
                x=x_trend,
                y=trendline(x_trend),
                mode='lines',
                name='Линия тренда',
                line=dict(color='grey', dash='dash')
            ))
        
        # Выделение выбранного пункта
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
        
        # Расчет корреляции
        if len(x) > 1:
            corr_coef = np.corrcoef(x, y)[0, 1]
            st.info(f"""
            **Коэффициент корреляции**: {corr_coef:.2f}
            - Близко к 1: Сильная прямая связь
            - Близко к 0: Нет связи
            - Близко к -1: Сильная обратная связь
            """)
        else:
            st.warning("Недостаточно данных для расчета корреляции")
            
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
