import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score, f1_score
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

class SkinTypeClusteringService:
    def __init__(self):
        self.model_directory = "model"
        os.makedirs(self.model_directory, exist_ok=True)
        self.stop_words_list = [
            'water', 'aqua', 'eau', 'glycerin', 'phenoxyethanol', 'butylene', 'glycol', 
            'dimethicone', 'sodium', 'chloride', 'citric', 'acid', 'xanthan', 'gum', 
            'disodium', 'edta', 'benzoate', 'potassium', 'sorbate', 'ethylhexylglycerin'
        ]
        
    def findClusterForSkinType(self, dataframe):
        print("[CLUSTERING] Starting clustering process (Target: Silhouette > 0.5)")
        
        vectorizer = TfidfVectorizer(
            max_df=0.5,
            min_df=2,
            stop_words=self.stop_words_list,
            sublinear_tf=True
        )
        tfidf_matrix = vectorizer.fit_transform(dataframe['clean_ingredients'].fillna(''))

        dense_matrix = tfidf_matrix.toarray()
        principal_component_analysis = PCA(n_components=2, random_state=42) 
        pca_features = principal_component_analysis.fit_transform(dense_matrix)
        
        best_k_clusters = 4
        best_silhouette_score = -1
        
        for test_k in range(3, 6):
            kmeans_test_model = KMeans(n_clusters=test_k, n_init=20, random_state=42)
            predicted_labels = kmeans_test_model.fit_predict(pca_features)
            current_silhouette = silhouette_score(pca_features, predicted_labels)
            
            if current_silhouette > best_silhouette_score:
                best_silhouette_score = current_silhouette
                best_k_clusters = test_k

        final_kmeans_model = KMeans(n_clusters=best_k_clusters, n_init=100, random_state=42)
        final_labels = final_kmeans_model.fit_predict(pca_features)

        clustering_plot_buffer = self.generate_cluster_plot(pca_features, final_labels)
        
        final_silhouette_value = float(best_silhouette_score)
        sum_of_squared_errors = float(final_kmeans_model.inertia_)
        
        print(f"[CLUSTERING] Completed. Selected K: {best_k_clusters}, Silhouette Score: {final_silhouette_value:.5f}")

        dataframe_result = dataframe.copy()
        dataframe_result['cluster'] = final_labels
        cluster_profile = {}
        actual_labels_list = []
        predicted_labels_list = []
        feature_names = np.array(vectorizer.get_feature_names_out())

        for cluster_index in range(best_k_clusters):
            cluster_indices = dataframe_result[dataframe_result['cluster'] == cluster_index].index
            cluster_products = dataframe_result.loc[cluster_indices]
            
            valid_rows = cluster_products[cluster_products['skin_type'].notna() & (cluster_products['skin_type'] != '[]') & (cluster_products['skin_type'] != 'unknown')]
            
            all_skin_types_in_cluster = []
            for skin_type_entry in valid_rows['skin_type']:
                cleaned = str(skin_type_entry).replace('[', '').replace(']', '').replace("'", "").replace('"', '')
                all_skin_types_in_cluster.extend([t.strip() for t in cleaned.split(',') if t.strip()])

            common_types = Counter(all_skin_types_in_cluster).most_common(1)
            dominant_label = common_types[0][0] if common_types else "combination"

            for idx, row in valid_rows.iterrows():
                actual_val_string = str(row['skin_type']).lower()
                if dominant_label in actual_val_string:
                    actual_labels_list.append(dominant_label)
                else:
                    actual_labels_list.append(actual_val_string.split(',')[0].strip())
                
                predicted_labels_list.append(dominant_label)

            cluster_tfidf_subset = tfidf_matrix[cluster_indices]
            mean_tfidf_values = np.asarray(cluster_tfidf_subset.mean(axis=0)).flatten()
            top_ingredient_indices = mean_tfidf_values.argsort()[-15:][::-1]
            
            cluster_profile[cluster_index] = {
                "dominant_skin_types": [dominant_label],
                "key_ingredients": feature_names[top_ingredient_indices].tolist()
            }
        
        final_accuracy = float(accuracy_score(actual_labels_list, predicted_labels_list)) if actual_labels_list else 0.0
        final_f1_score = float(f1_score(actual_labels_list, predicted_labels_list, average='weighted')) if actual_labels_list else 0.0

        return dataframe_result, cluster_profile, final_silhouette_value, sum_of_squared_errors, final_accuracy, final_f1_score, clustering_plot_buffer
    
    def generate_cluster_plot(self, pca_features, labels):
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x=pca_features[:, 0], 
            y=pca_features[:, 1], 
            hue=labels, 
            palette='viridis', 
            s=60, 
            alpha=0.7
        )
        plt.title('Product Clustering Visualization (PCA 2D)', fontsize=15)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Cluster')
        
        plot_buffer = BytesIO()
        plt.savefig(plot_buffer, format='png', bbox_inches='tight')
        plt.close()
        plot_buffer.seek(0)
        return plot_buffer