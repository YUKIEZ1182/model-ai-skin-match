import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load your dataset (adjust the file path as needed)
df = pd.read_csv("data/cosmetics_cleaned.csv")

# Optional: Combine ingredients into a single string for each product
# (Replace commas with spaces so that CountVectorizer can treat each ingredient as a token)
df['ingredients_text'] = df['clean_ingredients'].apply(lambda x: x.replace(",", " "))

# Create a one-hot encoded matrix for ingredients
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), binary=True)
X = vectorizer.fit_transform(df['ingredients_text'])

# Decide on the number of clusters (k). Here we use k=3 as an example.
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X)

# Add the cluster labels to your dataframe
df['cluster'] = clusters

# Compute the inertia (sum of squared errors)
inertia = kmeans.inertia_
print("Inertia (Sum of squared errors):", inertia)

# Compute the silhouette score (the higher, the better, ranges from -1 to 1)
silhouette = silhouette_score(X, clusters)
print("Silhouette Score:", silhouette)

# Optionally, view how many products fall into each cluster
print("Products per cluster:")
print(df['cluster'].value_counts())

# Optional: Visualize the inertia for different k to find the elbow
inertias = []
k_range = range(1, 10)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(k_range, inertias, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Sum of Squared Errors)")
plt.title("Elbow Method For Optimal k")
plt.show()
