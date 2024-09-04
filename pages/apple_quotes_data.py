import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Котировки акций Apple")

ticker = "AAPL"
start_date = st.date_input("Выберите дату начала", pd.to_datetime("2024-01-01"))
end_date = st.date_input("Выберите дату окончания", pd.to_datetime("today"))

data = yf.download(ticker, start=start_date, end=end_date)

if not data.empty:
    st.write("Данные по котировкам за выбранный период:")
    st.dataframe(data)

    # Построение графика
    st.line_chart(data['Close'], use_container_width=True)

    # Суммарные статистики
    st.write(f"Средняя цена закрытия за указанный период: {data['Close'].mean()}")
    st.write(f"Максимальная цена закрытия за указанный период: {data['Close'].max()}")
    st.write(f"Минимальная цена закрытия за указанный период: {data['Close'].min()}")
else:
    st.write("Вы не ввели данные для отображения котировок.")