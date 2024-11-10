# Імпорт необхідних бібліотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
import joblib


# Частина 1
# Завантаження датасету
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target  # Додаємо цільову змінну

# Описова статистика для всіх ознак
desc_stats = df.describe()
print("Описова статистика для всіх ознак:\n", desc_stats)

# Перевірка на пропущені значення
missing_values = df.isnull().sum()
print("Пропущені значення:\n", missing_values)

# Визначення типів даних кожної колонки
data_types = df.dtypes
print("Типи даних:\n", data_types)

# Побудова гістограм
df.hist(bins=30, figsize=(15, 10), layout=(3, 3))
plt.suptitle("Гістограми розподілу для кожної ознаки")
plt.show()

# Побудова boxplot для кожної ознаки
plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[column])
    plt.title(column)
plt.tight_layout()
plt.show()

# Побудова кореляційної матриці
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Теплова карта кореляційної матриці")
plt.show()

# Побудова scatter plots для цільової змінної та ознак
plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 3, i)
    plt.scatter(df[column], df['MedHouseVal'], alpha=0.5)
    plt.xlabel(column)
    plt.ylabel('MedHouseVal')
    plt.title(f'{column} vs MedHouseVal')
plt.tight_layout()
plt.show()

# Частина 2
# Визначення ознак та цільової змінної
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Розділення даних на тренувальну (80%) та тестову (20%) вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабування ознак
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Збереження скейлера для подальшого використання
joblib.dump(scaler, 'scaler.pkl')
print("Скейлер збережено як 'scaler.pkl'")

# Частина 3
# Визначення найбільш корельованої ознаки
correlations = df.corr()
top_feature = correlations['MedHouseVal'].abs().sort_values(ascending=False).index[1]  # Найбільша кореляція з MedHouseVal
print(f"Найбільш корельована ознака з MedHouseVal: {top_feature}")

# Підготовка даних для простої лінійної регресії
X_simple_train = X_train[[top_feature]]
X_simple_test = X_test[[top_feature]]

# Створення та навчання моделі
simple_model = LinearRegression()
simple_model.fit(X_simple_train, y_train)

# Прогнозування та візуалізація результатів
y_pred_simple = simple_model.predict(X_simple_test)
plt.scatter(X_simple_test, y_test, color="blue", alpha=0.5, label="Реальні значення")
plt.plot(X_simple_test, y_pred_simple, color="red", label="Прогноз")
plt.xlabel(top_feature)
plt.ylabel("MedHouseVal")
plt.title("Проста лінійна регресія")
plt.legend()
plt.show()

# Обчислення метрик якості
mse_simple = mean_squared_error(y_test, y_pred_simple)
r2_simple = r2_score(y_test, y_pred_simple)
print(f"Метрики простої лінійної регресії - MSE: {mse_simple}, R²: {r2_simple}")

# Підготовка та навчання моделі
multiple_model = LinearRegression()
multiple_model.fit(X_train_scaled, y_train)

# Прогнозування на тестовій вибірці
y_pred_multiple = multiple_model.predict(X_test_scaled)

# Аналіз коефіцієнтів
coefficients = pd.DataFrame(multiple_model.coef_, index=X.columns, columns=['Coefficient'])
print("Коефіцієнти моделі:\n", coefficients)

# Обчислення метрик якості
mse_multiple = mean_squared_error(y_test, y_pred_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)
print(f"Метрики множинної лінійної регресії - MSE: {mse_multiple}, R²: {r2_multiple}")

# Вибір ознак за допомогою моделі Lasso
selector = SelectFromModel(LassoCV(cv=5))
selector.fit(X_train_scaled, y_train)

# Відфільтровані ознаки
selected_features = X.columns[(selector.get_support())]
print(f"Відібрані ознаки: {selected_features}")

# Створення оптимізованої моделі з відібраними ознаками
X_train_opt = selector.transform(X_train_scaled)
X_test_opt = selector.transform(X_test_scaled)

# Навчання моделі
optimized_model = LinearRegression()
optimized_model.fit(X_train_opt, y_train)

# Прогнозування та обчислення метрик якості
y_pred_optimized = optimized_model.predict(X_test_opt)
mse_optimized = mean_squared_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)
print(f"Метрики оптимізованої моделі - MSE: {mse_optimized}, R²: {r2_optimized}")


# Частина 4
# Функція для розрахунку Adjusted R²
def adjusted_r2(r2, n, k):
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))

# Кількість спостережень і кількість ознак
n = X_test.shape[0]
k_simple = 1  # Для простої моделі - одна ознака
k_multiple = X_train.shape[1]  # Для множинної - всі ознаки
k_optimized = X_train_opt.shape[1]  # Для оптимізованої - відібрані ознаки

# Метрики для простої моделі
rmse_simple = np.sqrt(mse_simple)
adj_r2_simple = adjusted_r2(r2_simple, n, k_simple)

# Метрики для множинної моделі
rmse_multiple = np.sqrt(mse_multiple)
adj_r2_multiple = adjusted_r2(r2_multiple, n, k_multiple)

# Метрики для оптимізованої моделі
rmse_optimized = np.sqrt(mse_optimized)
adj_r2_optimized = adjusted_r2(r2_optimized, n, k_optimized)

# Порівняння метрик
metrics = pd.DataFrame({
    'Model': ['Simple Linear Regression', 'Multiple Linear Regression', 'Optimized Linear Regression'],
    'MSE': [mse_simple, mse_multiple, mse_optimized],
    'RMSE': [rmse_simple, rmse_multiple, rmse_optimized],
    'R²': [r2_simple, r2_multiple, r2_optimized],
    'Adjusted R²': [adj_r2_simple, adj_r2_multiple, adj_r2_optimized]
})

print(metrics)

# Передбачені значення для всіх моделей
predictions = {
    'Simple Model': y_pred_simple,
    'Multiple Model': y_pred_multiple,
    'Optimized Model': y_pred_optimized
}

# Побудова графіку
plt.figure(figsize=(15, 5))
for i, (label, y_pred) in enumerate(predictions.items(), 1):
    plt.subplot(1, 3, i)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Реальні значення")
    plt.ylabel("Передбачені значення")
    plt.title(f"{label}: Передбачені vs Реальні")
plt.tight_layout()
plt.show()

# Графік залишків для всіх моделей
plt.figure(figsize=(15, 5))
for i, (label, y_pred) in enumerate(predictions.items(), 1):
    residuals = y_test - y_pred
    plt.subplot(1, 3, i)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Передбачені значення")
    plt.ylabel("Залишки")
    plt.title(f"{label}: Залишки")
plt.tight_layout()
plt.show()

# Розподіл залишків для всіх моделей
plt.figure(figsize=(15, 5))
for i, (label, y_pred) in enumerate(predictions.items(), 1):
    residuals = y_test - y_pred
    plt.subplot(1, 3, i)
    sns.histplot(residuals, kde=True)
    plt.xlabel("Залишки")
    plt.title(f"{label}: Розподіл залишків")
plt.tight_layout()
plt.show()

#Частина 5
# Завантаження збереженого скейлера та моделі
scaler = joblib.load('scaler.pkl')
optimized_model = LinearRegression()  # Повинна бути збережена оптимізована модель
optimized_model.fit(X_train_opt, y_train)  # Навчимо на оптимізованих ознаках

def predict_house_price(features):
    # Перевірка на наявність необхідних характеристик
    required_features = list(selected_features)
    if not all(feature in features for feature in required_features):
        raise ValueError(f"Будь ласка, вкажіть усі необхідні характеристики: {required_features}")
    
    # Формування даних
    input_data = np.array([[features[feature] for feature in required_features]])
    
    # Масштабування даних
    input_data_scaled = scaler.transform(input_data)
    
    # Прогноз
    predicted_price = optimized_model.predict(input_data_scaled)
    
    return predicted_price[0]

# Приклад використання функції
sample_features = {
    'MedInc': 5.0,
    'HouseAge': 15,
    'AveRooms': 6,
    'AveBedrms': 1,
    'Population': 800,
    'AveOccup': 3,
    'Latitude': 34.0,
    'Longitude': -118.0
}
print("Прогнозована ціна:", predict_house_price(sample_features))