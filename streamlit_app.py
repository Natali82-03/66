import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import chardet

# Конфигурация страницы (должна быть первой!)
st.set_page_config(layout="wide")

# --- Загрузка данных ---
@st.cache_data
def load_data(file_name):
    with open(file_name, 'rb') as f:
        result = chardet.detect(f.read(10000))
    try:
        df = pd.read_csv(file_name, sep=';', encoding=result['encoding'])
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_name, sep=';', encoding='utf-8')
        except:
            df = pd.read_csv(file_name, sep=';', encoding='cp1251')
    
    # Очистка данных
    df = df.rename(columns=lambda x: x.strip())
    if 'Наименование муниципального образования' in df.columns:
        df = df.rename(columns={'Наименование муниципального образования': 'Name'})
    df['Name'] = df['Name'].str.strip()
    return df

# Загрузка всех файлов
try:
    ch_1_6 = load_data('Ch_1_6.csv')      # Дети 1-6 лет
    ch_3_18 = load_data('Ch_3_18.csv')    # Дети 3-18 лет
    ch_5_18 = load_data('Ch_5_18.csv')    # Дети 5-18 лет
    pop_3_79 = load_data('Pop_3_79.csv')  # Население 3-79 лет
    rpop = load_data('RPop.csv')          # Среднегодовая численность населения
except Exception as e:
    st.error(f"Ошибка загрузки данных: {str(e)}")
    st.stop()

# Словарь тем (название: (датафрейм, описание, цвет))
data_dict = {
    "Дети 1-6 лет": (ch_1_6, "Численность детей 1-6 лет", "skyblue"),
    "Дети 3-18 лет": (ch_3_18, "Численность детей 3-18 лет", "salmon"),
    "Дети 5-18 лет": (ch_5_18, "Численность детей 5-18 лет", "gold"),
    "Население 3-79 лет": (pop_3_79, "Численность населения 3-79 лет", "lightgreen"),
    "Среднегодовая численность": (rpop, "Среднегодовая численность постоянного населения", "violet")
}

# --- Боковая панель ---
with st.sidebar:
    st.title("Настройки анализа")
    
    # Выбор населенного пункта
    all_locations = ch_1_6['Name'].unique()
    selected_location = st.selectbox("Населённый пункт:", all_locations, index=0)
    
    # Выбор тем (можно несколько)
    selected_topics = st.multiselect(
        "Категории населения:",
        list(data_dict.keys()),
        default=["Дети 1-6 лет", "Среднегодовая численность"]
    )
    
    if not selected_topics:
        st.warning("Выберите хотя бы одну категорию!")
        st.stop()
    
    # Фиксированные годы (2019-2024)
    year_columns = [str(year) for year in range(2019, 2025)]
    
    # Выбор диапазона лет
    year_range = st.slider(
        "Диапазон лет:",
        min_value=2019,
        max_value=2024,
        value=(2019, 2024)
    )
    selected_years = [str(year) for year in range(year_range[0], year_range[1]+1)]

# --- Основной интерфейс ---
st.title(f"Демография Орловской области: {selected_location}")

# 1. Линейный график
st.subheader("Динамика численности")
fig_line, ax_line = plt.subplots(figsize=(12, 5))
for topic in selected_topics:
    df, label, color = data_dict[topic]
    location_data = df[df['Name'] == selected_location]
    if not location_data.empty:
        ax_line.plot(
            selected_years,
            location_data[selected_years].values.flatten(),
            label=label, color=color, marker='o', linewidth=2
        )
ax_line.set_xlabel("Год")
ax_line.set_ylabel("Численность (чел.)")
ax_line.legend()
ax_line.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig_line)

# 2. Столбчатая диаграмма
st.subheader("Сравнение по годам")
fig_bar, ax_bar = plt.subplots(figsize=(12, 6))

bar_width = 0.8 / len(selected_topics)  # Ширина столбцов
opacity = 0.8

for i, topic in enumerate(selected_topics):
    df, label, color = data_dict[topic]
    location_data = df[df['Name'] == selected_location]
    if not location_data.empty:
        values = location_data[selected_years].values.flatten()
        positions = [x + i * bar_width for x in range(len(selected_years))]
        ax_bar.bar(
            positions, values, bar_width,
            alpha=opacity, color=color, label=label
        )

ax_bar.set_xticks([x + bar_width * (len(selected_topics)-1)/2 for x in range(len(selected_years))])
ax_bar.set_xticklabels(selected_years)
ax_bar.set_xlabel("Год")
ax_bar.set_ylabel("Численность (чел.)")
ax_bar.legend(bbox_to_anchor=(1.05, 1))
ax_bar.grid(True, axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig_bar)

# 3. Таблицы с данными
st.subheader("Детальные данные")
for topic in selected_topics:
    df, label, _ = data_dict[topic]
    st.markdown(f"**{label}**")
    st.dataframe(
        df[df['Name'] == selected_location][['Name'] + selected_years],
        use_container_width=True,
        height=120
    )