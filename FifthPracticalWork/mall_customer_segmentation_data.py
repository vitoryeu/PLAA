import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Частина 1
# Завантаження датасету
data = pd.read_csv('Mall_Customers.csv')

# Перевірка наявності пропущених значень у кожному стовпці
missing_values = data.isnull().sum()
print("Пропущені значення у кожному стовпці:")
print(missing_values)

# Налаштування розміщення підграфіків
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Гістограми розподілу для кожної змінної', fontsize=16)

# Гістограма для віку
axes[0, 0].hist(data['Age'], bins=15, edgecolor='black')
axes[0, 0].set_title('Розподіл віку')
axes[0, 0].set_xlabel('Вік')
axes[0, 0].set_ylabel('Кількість')

# Гістограма для річного доходу
axes[0, 1].hist(data['Annual Income (k$)'], bins=15, edgecolor='black')
axes[0, 1].set_title('Розподіл річного доходу')
axes[0, 1].set_xlabel('Річний дохід (тис. $)')
axes[0, 1].set_ylabel('Кількість')

# Гістограма для оцінки витрат
axes[1, 0].hist(data['Spending Score (1-100)'], bins=15, edgecolor='black')
axes[1, 0].set_title('Розподіл оцінки витрат')
axes[1, 0].set_xlabel('Оцінка витрат (1-100)')
axes[1, 0].set_ylabel('Кількість')

# Стовпчикова діаграма для розподілу статі
gender_counts = data['Gender'].value_counts()
axes[1, 1].bar(gender_counts.index, gender_counts.values, color=['skyblue', 'lightcoral'], edgecolor='black')
axes[1, 1].set_title('Розподіл статі')
axes[1, 1].set_xlabel('Стать')
axes[1, 1].set_ylabel('Кількість')

# Налаштування відступів
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Розрахунок основних статистичних показників для датасету
statistical_summary = data.describe(include='all')
print(statistical_summary)

# Вибір числових стовпців для стандартизації
numerical_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
scaler = StandardScaler()

# Виконання стандартизації
standardized_data = scaler.fit_transform(data[numerical_columns])

# Перетворення стандартизованих даних назад у DataFrame для зручності
standardized_df = pd.DataFrame(standardized_data, columns=numerical_columns)
print(standardized_df.head())

# Частина 2
# Визначення діапазону для кількості кластерів
k_values = range(1, 11)
inertia_values = []

# Розрахунок інерції для кожного значення k за допомогою KMeans
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(standardized_df)
    inertia_values.append(kmeans.inertia_)

# Побудова графіка методу ліктя
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.title('Залежність інерції від кількості кластерів (Метод ліктя)')
plt.xlabel('Кількість кластерів (k)')
plt.ylabel('Інерція')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Розрахунок коефіцієнтів силуету для різної кількості кластерів
silhouette_scores = []

# Обчислення коефіцієнта силуету для кожного k (від 2 до 10)
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(standardized_df)
    score = silhouette_score(standardized_df, cluster_labels)
    silhouette_scores.append(score)

# Побудова графіка коефіцієнта силуету
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Залежність коефіцієнта силуету від кількості кластерів')
plt.xlabel('Кількість кластерів (k)')
plt.ylabel('Коефіцієнт силуету')
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()

#Частина 3
# Виконання кластеризації методом K-means з 5 кластерами
kmeans = KMeans(n_clusters=5, random_state=0)
data['Cluster'] = kmeans.fit_predict(standardized_df)

# Побудова scatter plot для візуалізації кластерів за віком та річним доходом
plt.figure(figsize=(10, 6))
plt.scatter(data['Age'], data['Annual Income (k$)'], 
            c=data['Cluster'], cmap='viridis', s=50, alpha=0.7, edgecolor='k')
plt.title('Кластеризація клієнтів (K-means) за віком і річним доходом')
plt.xlabel('Вік')
plt.ylabel('Річний дохід (тис. $)')
plt.colorbar(label='Номер кластера')
plt.grid(True)
plt.show()

# Витягування центроїдів кластерів
centroids = kmeans.cluster_centers_

# Побудова scatter plot для точок кластерів та центроїдів
plt.figure(figsize=(10, 6))
plt.scatter(data['Age'], data['Annual Income (k$)'], 
            c=data['Cluster'], cmap='viridis', s=50, alpha=0.7, edgecolor='k', label='Клієнти')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Центроїди')
plt.title('Кластеризація клієнтів (K-means) з центроїдами')
plt.xlabel('Вік')
plt.ylabel('Річний дохід (тис. $)')
plt.colorbar(label='Номер кластера')
plt.legend()
plt.grid(True)
plt.show()

# Розрахунок середніх значень показників для кожного кластера
cluster_means = data.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(cluster_means)