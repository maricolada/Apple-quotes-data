import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Название
#Описание
st.title('Анализ чаевых в ресторане')
st.write('В этом приложении вы можете получить визуализацию данных по чаевым в вашем ресторане.')


## Шаг 1. Загрузка csv файла

uploaded_file = st.file_uploader('Загрузите CSV файл, чтобы начать работу. Для корректной работы приложения убедитесь, что ваша таблица содержит столбцы по итоговой сумме счёта (total_bill), по чаевым по соответствующему счёту (tip), данные о поле (sex) в формате Female / Male, о курящих/некурящих (smoker) в формате Yes / No, дне недели (day), времени приема пищи (time) в формате Dinner / Lunch, а также о количестве позиций в заказе (size).', type='csv')
df = pd.read_csv(uploaded_file)
st.write(df.head(10))

## Шаг 2. Добавляем столбец со временем заказа и рандомной датой

def random_dates(start, end, n=1):
    start_u = start.value // 10**9
    end_u = end.value // 10**9
    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')

start_date = pd.to_datetime('2023-01-01')
end_date = pd.to_datetime('2023-01-31')

# Добавляем столбец со рандомной датой при нажатии на кнопку

upd_df = pd.DataFrame()

if st.button('Смотреть аналитику'):
    
    upd_df = df.copy()
    upd_df['time_order'] = random_dates(start_date, end_date, len(upd_df))
    st.write(upd_df)

    csv = upd_df.to_csv(index=False)  # Преобразуем в CSV
        
    st.subheader("Динамика чаевых во времени")

    plt.figure(figsize=(10, 5))
    st.line_chart(data=upd_df, x='time_order', y='tip', color=None, width=None, height=None, use_container_width=True)
    plt.grid(True)
    plt.style.use('ggplot')


    st.subheader("Динамика итоговой суммы счёта")

    hist, bins = np.histogram(upd_df['total_bill'], bins=30)
    hist_df = pd.DataFrame({'Quantity': hist, 'Bins': (bins[:-1] + bins[1:]) / 2})
    st.bar_chart(hist_df.set_index('Bins'))
    

    st.subheader("Связь между суммой счёта и чаевыми")

    scatter_data = upd_df[['total_bill', 'tip']]
    st.scatter_chart(scatter_data.set_index('total_bill'))


    st.subheader("Связь между суммой счёта, чаевыми и их размером")

    scatter_data = upd_df[['tip', 'total_bill']] 
    st.scatter_chart(scatter_data)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=upd_df, x='total_bill', y='tip', size='size', sizes=(20, 100), hue='size', palette='viridis', legend=True)
    plt.title('Связь между суммой счёта, чаевыми и их размером')
    plt.xlabel('Сумма счёта')
    plt.ylabel('Чаевые')
    plt.grid(True)
    plt.style.use('ggplot')
    file_name = 'total_bills_tips_size.png'  
    plt.savefig(file_name)
    
    st.pyplot(plt)

    with open(file_name, 'rb') as file:
        st.download_button(
            label="Скачать график как PNG",
            data=file,
            file_name=file_name,
            mime='image/png'
            )

    st.subheader("Связь между днём недели и размером счёта")

    days_of_week = ['Thur', 'Fri', 'Sat', 'Sun']
    bill_by_day = upd_df.groupby('day')['total_bill'].mean().reset_index()
    bill_by_day['day'] = pd.Categorical(bill_by_day['day'], categories=days_of_week, ordered=True)
    bill_by_day = bill_by_day.sort_values('day')
    st.bar_chart(bill_by_day.set_index('day'))

   
    st.subheader("Связь между чаевыми, днями недели и полом")

    colors = ['blue' if x == 'Male' else 'red' for x in upd_df['sex']] 
    plt.figure(figsize=(15, 4))
    sns.scatterplot(data=upd_df, x='tip', y='day', c=colors, legend=True)
    plt.title('Связь между чаевыми, днями недели и полом')
    plt.xlabel('Чаевые')
    plt.ylabel('Дни')
    plt.style.use('ggplot')
    plt.grid(True)
    file_name = 'tips_days_gender.png'  
    plt.savefig(file_name)
    
    st.pyplot(plt)

    with open(file_name, 'rb') as file:
        st.download_button(
            label="Скачать график как PNG",
            data=file,
            file_name=file_name,
            mime='image/png'
            )

    st.subheader("Связь между временем приёма пищи и суммой всех счетов")

    plt.figure(figsize=(15, 3))
    sns.boxplot(data=upd_df, x='total_bill', y='time', hue='time', palette='inferno', legend=True)
    plt.title('Связь между временем приёма пищи и суммой всех счетов')
    plt.xlabel('Размер счёта')
    plt.ylabel('Время приема пищи')
    plt.legend()
    plt.grid(True)
    file_name = 'day_sum_bills.png'  
    plt.savefig(file_name)
    
    st.pyplot(plt)

    with open(file_name, 'rb') as file:
        st.download_button(
            label="Скачать график как PNG",
            data=file,
            file_name=file_name,
            mime='image/png'
            )
    
    st.subheader("Соотношение чаевых на ужин и обед")

    mean_tips = upd_df.groupby('time')['tip'].mean().reset_index()
    st.bar_chart(mean_tips.set_index('time'))

    plt.style.use('ggplot')

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(upd_df[upd_df['time'] == 'Dinner']['tip'], bins=10, kde=True, color='red')
    plt.title('Чаевые (Dinner)')
    plt.xlabel('Чаевые')
    plt.ylabel('Количество')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    sns.histplot(upd_df[upd_df['time'] == 'Lunch']['tip'], bins=10, kde=True, color='blue')
    plt.title('Чаевые (Lunch)')
    plt.xlabel('Чаевые')
    plt.ylabel('Количество')
    plt.grid(True)
    
    file_name = 'lunch_dinner_tips.png'  
    plt.savefig(file_name)
    
    st.pyplot(plt)

    with open(file_name, 'rb') as file:
        st.download_button(
            label="Скачать график как PNG",
            data=file,
            file_name=file_name,
            mime='image/png'
            )
    
    
    st.subheader("Соотношение размера счёта и чаевых в зависимости от курящих и некурящих посетителей")

    plt.style.use('ggplot')
    plt.subplot(1, 2, 1)
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=upd_df[upd_df['sex'] == 'Male'], 
                x='total_bill', 
                y='tip', 
                hue='smoker', 
                style='smoker', 
                markers={'Yes': 'o', 'No': 'X'}, 
                palette='inferno', 
                s=80)
    plt.title('Мужчины')
    plt.xlabel('Размер счёта')
    plt.ylabel('Чаевые')
    plt.legend(title='(Не)курящий')
    file_name = 'tips_males_smokers.png'  
    plt.savefig(file_name)
    
    st.pyplot(plt)

    with open(file_name, 'rb') as file:
        st.download_button(
            label="Скачать график как PNG",
            data=file,
            file_name=file_name,
            mime='image/png'
            )
    
    plt.subplot(1, 2, 2)
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=upd_df[upd_df['sex'] == 'Female'], 
                x='total_bill', 
                y='tip', 
                hue='smoker', 
                style='smoker', 
                markers={'Yes': 'o', 'No': 'X'}, 
                palette='inferno', 
                s=80)
    plt.title('Женщины')
    plt.xlabel('Размер счёта')
    plt.ylabel('Чаевые')
    plt.legend(title='(Не)курящая')

    file_name = 'tips_females_smokers.png'  
    plt.savefig(file_name)
    
    st.pyplot(plt)

    with open(file_name, 'rb') as file:
        st.download_button(
            label="Скачать график как PNG",
            data=file,
            file_name=file_name,
            mime='image/png'
            )
    
    st.subheader("Тепловая карта зависимостей численных переменных")

    numerical_data = upd_df.select_dtypes(include=[np.number])
    correlation_matrix = numerical_data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Тепловая карта зависимостей численных переменных')
    file_name = 'heat_map_numerical.png'  
    plt.savefig(file_name)
    
    st.pyplot(plt)

    with open(file_name, 'rb') as file:
        st.download_button(
            label="Скачать график как PNG",
            data=file,
            file_name=file_name,
            mime='image/png'
            )





