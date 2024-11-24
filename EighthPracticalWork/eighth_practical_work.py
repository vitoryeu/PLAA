import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Завантаження даних акцій за останній рік (компанія Apple)
ticker = "AAPL"
data = yf.Ticker(ticker).history(period="1y")

# Перевірка наявності пропущених значень
missing_values = data.isnull().sum()
print("Пропущені значення в даних:")
print(missing_values)

# Графік зміни ціни закриття
plt.figure(figsize=(10, 5))
plt.plot(data['Close'], label='Ціна закриття')
plt.title('Зміна ціни закриття за останній рік')
plt.xlabel('Дата')
plt.ylabel('Ціна ($)')
plt.legend()
plt.grid()
plt.show()

# Базова описова статистика
print("Базова описова статистика:")
print(data.describe())

# Декомпозиція часового ряду
result = seasonal_decompose(data['Close'], model='additive', period=30)  # 30 днів - місячний період

# Візуалізація результатів декомпозиції
plt.figure(figsize=(10, 8))

# Тренд
plt.subplot(4, 1, 1)
plt.plot(result.trend, label='Тренд', color='blue')
plt.title('Тренд')
plt.legend()

# Сезонність
plt.subplot(4, 1, 2)
plt.plot(result.seasonal, label='Сезонність', color='orange')
plt.title('Сезонність')
plt.legend()

# Випадкова компонента
plt.subplot(4, 1, 3)
plt.plot(result.resid, label='Випадкова компонента', color='green')
plt.title('Випадкова компонента')
plt.legend()

# Оригінальний часовий ряд
plt.subplot(4, 1, 4)
plt.plot(data['Close'], label='Оригінальний ряд', color='black')
plt.title('Оригінальний часовий ряд')
plt.legend()

plt.tight_layout()
plt.show()

# Аналіз
print("Огляд тренду:")
print(result.trend.dropna().head())

print("Огляд сезонної складової:")
print(result.seasonal.head())

print("Огляд випадкової компоненти:")
print(result.resid.dropna().head())


# Прості ковзні середні (SMA)
data['SMA_7'] = data['Close'].rolling(window=7).mean()
data['SMA_30'] = data['Close'].rolling(window=30).mean()

# Індекс відносної сили (RSI)
def calculate_rsi(data, period=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data)

# Волатильність (30-денна стандартне відхилення)
data['Volatility_30'] = data['Close'].rolling(window=30).std()

# Визначення рівнів підтримки та опору
support_level = data['Close'].min()
resistance_level = data['Close'].max()

# Побудова графіків
plt.figure(figsize=(14, 10))

# Графік цін з SMA
plt.subplot(3, 1, 1)
plt.plot(data['Close'], label='Ціна закриття', color='black')
plt.plot(data['SMA_7'], label='7-денне SMA', color='blue')
plt.plot(data['SMA_30'], label='30-денне SMA', color='red')
plt.axhline(y=support_level, color='green', linestyle='--', label='Рівень підтримки')
plt.axhline(y=resistance_level, color='orange', linestyle='--', label='Рівень опору')
plt.title('Ціни закриття та SMA')
plt.legend()

# RSI
plt.subplot(3, 1, 2)
plt.plot(data['RSI'], label='RSI (14)', color='purple')
plt.axhline(70, color='red', linestyle='--', label='Перекупленість')
plt.axhline(30, color='green', linestyle='--', label='Перепроданість')
plt.title('RSI')
plt.legend()

# Волатильність
plt.subplot(3, 1, 3)
plt.plot(data['Volatility_30'], label='30-денна Волатильність', color='orange')
plt.title('Волатильність')
plt.legend()

plt.tight_layout()
plt.show()

# Аналіз точок перетину
crossing_points = data[(data['SMA_7'] > data['SMA_30']) & (data['SMA_7'].shift(1) <= data['SMA_30'].shift(1))]
print("Точки перетину (сигнали купівлі):")
print(crossing_points[['Close']])

crossing_points = data[(data['SMA_7'] < data['SMA_30']) & (data['SMA_7'].shift(1) >= data['SMA_30'].shift(1))]
print("Точки перетину (сигнали продажу):")
print(crossing_points[['Close']])

# Розділення даних
train_size = int(len(data) * 0.8)
train, test = data['Close'][:train_size], data['Close'][train_size:]

# Експоненційне згладжування
model_exp = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=30)
fit_exp = model_exp.fit()
forecast_exp = fit_exp.forecast(len(test))

# ARIMA-модель
model_arima = ARIMA(train, order=(1, 1, 1))  # Параметри p, d, q можуть бути змінені
fit_arima = model_arima.fit()
forecast_arima = fit_arima.forecast(steps=len(test))

# Оцінка якості прогнозу
mse_exp = mean_squared_error(test, forecast_exp)
mae_exp = mean_absolute_error(test, forecast_exp)

mse_arima = mean_squared_error(test, forecast_arima)
mae_arima = mean_absolute_error(test, forecast_arima)

print("Експоненційне згладжування:")
print(f"MSE: {mse_exp:.2f}, MAE: {mae_exp:.2f}")

print("ARIMA:")
print(f"MSE: {mse_arima:.2f}, MAE: {mae_arima:.2f}")

# Візуалізація результатів
plt.figure(figsize=(12, 6))

plt.plot(data['Close'], label='Фактичні дані', color='black')
plt.plot(test.index, forecast_exp, label='Експоненційне згладжування', color='blue')
plt.plot(test.index, forecast_arima, label='ARIMA', color='red')

plt.title("Прогнозування")
plt.xlabel("Дата")
plt.ylabel("Ціна закриття")
plt.legend()
plt.grid()
plt.show()