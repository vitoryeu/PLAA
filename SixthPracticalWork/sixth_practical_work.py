import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Підготовка даних
# Завантаження датасету
mall_data = pd.read_csv('Mall_Customers.csv')

# Перевірка наявності пропущених значень у кожному стовпці
missing_values = mall_data.isnull().sum()
print("Пропущені значення у кожному стовпці:")
print(missing_values)

# Налаштування розміщення підграфіків
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Гістограми розподілу для кожної змінної', fontsize=16)

# Гістограма для віку
axes[0, 0].hist(mall_data['Age'], bins=15, edgecolor='black')
axes[0, 0].set_title('Розподіл віку')
axes[0, 0].set_xlabel('Вік')
axes[0, 0].set_ylabel('Кількість')

# Гістограма для річного доходу
axes[0, 1].hist(mall_data['Annual Income (k$)'], bins=15, edgecolor='black')
axes[0, 1].set_title('Розподіл річного доходу')
axes[0, 1].set_xlabel('Річний дохід (тис. $)')
axes[0, 1].set_ylabel('Кількість')

# Гістограма для оцінки витрат
axes[1, 0].hist(mall_data['Spending Score (1-100)'], bins=15, edgecolor='black')
axes[1, 0].set_title('Розподіл оцінки витрат')
axes[1, 0].set_xlabel('Оцінка витрат (1-100)')
axes[1, 0].set_ylabel('Кількість')

# Стовпчикова діаграма для розподілу статі
gender_counts = mall_data['Gender'].value_counts()
axes[1, 1].bar(gender_counts.index, gender_counts.values, color=['skyblue', 'lightcoral'], edgecolor='black')
axes[1, 1].set_title('Розподіл статі')
axes[1, 1].set_xlabel('Стать')
axes[1, 1].set_ylabel('Кількість')

# Налаштування відступів
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Розрахунок основних статистичних показників для датасету
statistical_summary = mall_data.describe(include='all')
print(statistical_summary)

# Вибір числових стовпців для стандартизації
numerical_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
scaler = StandardScaler()

# Виконання стандартизації
standardized_data = scaler.fit_transform(mall_data[numerical_columns])

# Перетворення стандартизованих даних назад у DataFrame для зручності
standardized_df = pd.DataFrame(standardized_data, columns=numerical_columns)
print(standardized_df.head())

# Кодування категоріальної змінної (Gender) за допомогою one-hot кодування
mall_data_encoded = pd.get_dummies(mall_data, columns=['Gender'], drop_first=True)

# Перегляд перших кількох рядків оброблених даних
print("\nПерші кілька рядків оброблених даних:")
print(mall_data_encoded.head())

# Застосування методу PCA (Principal Component Analysis):
# Завантаження та підготовка даних (без CustomerID)
numerical_data = mall_data_encoded.drop(columns=['CustomerID'])

# Стандартизація даних перед застосуванням PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Застосування PCA (зменшення до 2 головних компонентів для візуалізації)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

# Створення DataFrame з PCA компонентами
pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])

# Перегляд перших кількох рядків компонент PCA
print("Перші кілька рядків компонент PCA:")
print(pca_df.head())

# Створення PCA без визначення кількості компонент, щоб проаналізувати всі
pca_full = PCA()
pca_full.fit(scaled_data)

# Розрахунок поясненої дисперсії для кожної компоненти
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()

# Створення DataFrame для відображення поясненої дисперсії
variance_df = pd.DataFrame({
    'Component': range(1, len(explained_variance_ratio) + 1),
    'Explained Variance Ratio': explained_variance_ratio,
    'Cumulative Variance Ratio': cumulative_variance_ratio
})

# Виведення результатів
print("Пояснена дисперсія для кожної компоненти:")
print(variance_df)

# 2D візуалізація з використанням перших двох головних компонент
plt.figure(figsize=(10, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], alpha=0.7)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('2D Visualization of Data with PCA')
plt.show()

# 3D візуалізація з використанням перших трьох головних компонент
# Застосування PCA для отримання 3 компонент для 3D графіка
pca_3d = PCA(n_components=3)
pca_3d_components = pca_3d.fit_transform(scaled_data)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_3d_components[:, 0], pca_3d_components[:, 1], pca_3d_components[:, 2], alpha=0.7)
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
ax.set_title('3D Visualization of Data with PCA')
plt.show()


# Застосування t-SNE до масштабованих даних з 2 компонентами для візуалізації
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(scaled_data)

# Створення DataFrame для результатів t-SNE
tsne_df = pd.DataFrame(data=tsne_results, columns=['t-SNE1', 't-SNE2'])

# Візуалізація результатів t-SNE
plt.figure(figsize=(10, 6))
plt.scatter(tsne_df['t-SNE1'], tsne_df['t-SNE2'], alpha=0.7)
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.title('2D Visualization of Data with t-SNE')
plt.show()

# Експериментальні значення параметрів
perplexities = [5, 30, 50]
learning_rates = [10, 100, 500]

# Створення графіків для кожної комбінації параметрів
fig, axes = plt.subplots(len(perplexities), len(learning_rates), figsize=(15, 12))

for i, perplexity in enumerate(perplexities):
    for j, learning_rate in enumerate(learning_rates):
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
        tsne_results = tsne.fit_transform(scaled_data)
        
        # Побудова графіку
        ax = axes[i, j]
        ax.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7)
        ax.set_title(f"Perplexity: {perplexity}, LR: {learning_rate}")
        ax.set_xlabel('t-SNE1')
        ax.set_ylabel('t-SNE2')

plt.tight_layout()
plt.show()

# Застосування K-means до даних після PCA
kmeans_pca = KMeans(n_clusters=4, random_state=42)
kmeans_pca_labels = kmeans_pca.fit_predict(pca_components)

# Застосування K-means до даних після t-SNE
kmeans_tsne = KMeans(n_clusters=4, random_state=42)
kmeans_tsne_labels = kmeans_tsne.fit_predict(tsne_results)

# Відображення кластерів K-means на даних після PCA
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=kmeans_pca_labels, cmap='viridis', alpha=0.7)
plt.title("K-means Clustering on PCA-reduced Data")
plt.xlabel("PCA1")
plt.ylabel("PCA2")

# Відображення кластерів K-means на даних після t-SNE
plt.subplot(1, 2, 2)
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=kmeans_tsne_labels, cmap='viridis', alpha=0.7)
plt.title("K-means Clustering on t-SNE-reduced Data")
plt.xlabel("t-SNE1")
plt.ylabel("t-SNE2")

plt.tight_layout()
plt.show()