import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score, f1_score
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

class SkinTypeClusteringService:
    def __init__(self):
        self.model_directory = "model"
        os.makedirs(self.model_directory, exist_ok=True)
        # Stop words list
        self.stop_words_list = [
            'water', 'aqua', 'eau', 'glycerin', 'phenoxyethanol', 'butylene', 'glycol', 
            'dimethicone', 'sodium', 'chloride', 'citric', 'acid', 'xanthan', 'gum', 
            'disodium', 'edta', 'benzoate', 'potassium', 'sorbate', 'ethylhexylglycerin',
            'fragrance', 'parfum', 'alcohol', 'denat'
        ]
        
    def findClusterForSkinType(self, dataframe):
        print("[CLUSTERING] Starting clustering process (With Coverage Guarantee)")
        
        # 1. Feature Extraction
        vectorizer = TfidfVectorizer(
            max_df=0.8,
            min_df=2,
            ngram_range=(1, 2),
            stop_words=self.stop_words_list,
            sublinear_tf=True
        )
        tfidf_matrix = vectorizer.fit_transform(dataframe['clean_ingredients'].fillna(''))

        dense_matrix = tfidf_matrix.toarray()
        principal_component_analysis = PCA(n_components=2, random_state=42) 
        pca_features = principal_component_analysis.fit_transform(dense_matrix)
        
        # 2. Find Best K
        best_k_clusters = 4
        best_silhouette_score = -1
        
        for test_k in range(4, 7):
            kmeans_test_model = KMeans(n_clusters=test_k, n_init=20, random_state=42)
            predicted_labels = kmeans_test_model.fit_predict(pca_features)
            current_silhouette = silhouette_score(pca_features, predicted_labels)
            
            if current_silhouette > best_silhouette_score:
                best_silhouette_score = current_silhouette
                best_k_clusters = test_k

        # 3. Final Train
        final_kmeans_model = KMeans(n_clusters=best_k_clusters, n_init=100, random_state=42)
        final_labels = final_kmeans_model.fit_predict(pca_features)

        clustering_plot_buffer = self.generate_cluster_plot(pca_features, final_labels)
        
        final_silhouette_value = float(best_silhouette_score)
        sum_of_squared_errors = float(final_kmeans_model.inertia_)
        
        print(f"[CLUSTERING] Completed. Selected K: {best_k_clusters}")

        dataframe_result = dataframe.copy()
        dataframe_result['cluster'] = final_labels
        
        cluster_profile = {}
        actual_labels_list = []
        predicted_labels_list = []
        feature_names = np.array(vectorizer.get_feature_names_out())

        # ตัวแปรสำหรับเก็บสถิติเพื่อทำ Coverage Guarantee
        cluster_skin_counts = defaultdict(Counter) # { cluster_id: Counter({'oily': 10, ...}) }
        all_found_skin_types = set()

        # 4. สร้าง Profile เบื้องต้น
        for cluster_index in range(best_k_clusters):
            cluster_indices = dataframe_result[dataframe_result['cluster'] == cluster_index].index
            cluster_products = dataframe_result.loc[cluster_indices]
            
            # ดึง Skin Type
            valid_rows = cluster_products[cluster_products['skin_type'].notna() & (cluster_products['skin_type'] != '[]')]
            
            all_skin_types_in_cluster = []
            for skin_type_entry in valid_rows['skin_type']:
                cleaned = str(skin_type_entry).replace('[', '').replace(']', '').replace("'", "").replace('"', '')
                types = [t.strip().lower() for t in cleaned.split(',') if t.strip()]
                all_skin_types_in_cluster.extend(types)
                all_found_skin_types.update(types)

            # เก็บสถิติ
            counts = Counter(all_skin_types_in_cluster)
            cluster_skin_counts[cluster_index] = counts
            total_count = sum(counts.values()) if counts else 1
            
            # Logic เดิม: เลือก Top 3 หรือ >15%
            dominant_labels = []
            sorted_types = counts.most_common()
            
            if sorted_types:
                dominant_labels.append(sorted_types[0][0]) # อันดับ 1 เสมอ
                for s_type, s_count in sorted_types[1:]:
                    percentage = s_count / total_count
                    if percentage > 0.15 or len(dominant_labels) < 3: 
                        if s_type not in dominant_labels:
                            dominant_labels.append(s_type)
            else:
                dominant_labels = ["combination"] 

            # หา Key Ingredients
            cluster_tfidf_subset = tfidf_matrix[cluster_indices]
            if cluster_tfidf_subset.shape[0] > 0:
                mean_tfidf_values = np.asarray(cluster_tfidf_subset.mean(axis=0)).flatten()
                top_ingredient_indices = mean_tfidf_values.argsort()[-10:][::-1]
                key_ingredients = feature_names[top_ingredient_indices].tolist()
            else:
                key_ingredients = []
            
            cluster_profile[cluster_index] = {
                "dominant_skin_types": dominant_labels,
                "key_ingredients": key_ingredients
            }

            # Accuracy Check logic (Simplified)
            primary_label = dominant_labels[0]
            for idx, row in valid_rows.iterrows():
                actual_val_string = str(row['skin_type']).lower()
                if primary_label in actual_val_string:
                    actual_labels_list.append(primary_label)
                else:
                    actual_labels_list.append(actual_val_string.split(',')[0].strip())
                predicted_labels_list.append(primary_label)

        # 5. ✅ Global Coverage Guarantee (ส่วนสำคัญที่เพิ่มเข้ามา)
        # ตรวจสอบว่า target_types ทั้งหมด ถูกระบุลงใน cluster ไหนหรือยัง
        target_types = ['oily', 'dry', 'combination', 'sensitive']
        
        for target in target_types:
            # เช็คว่า target นี้มีอยู่ใน dataset ไหม
            if target not in all_found_skin_types:
                continue # ถ้าไม่มีใน data เลยก็ข้าม

            # เช็คว่า target นี้ไปโผล่ใน dominant_skin_types ของกลุ่มไหนบ้างหรือยัง
            is_covered = False
            for c_id, profile in cluster_profile.items():
                if target in profile['dominant_skin_types']:
                    is_covered = True
                    break
            
            # ถ้ายังไม่มีกลุ่มไหนรับเป็น dominant เลย -> บังคับยัดใส่กลุ่มที่มี target นี้เยอะที่สุด
            if not is_covered:
                print(f"[CLUSTERING] Force assigning '{target}' to best matching cluster...")
                best_cluster_id = -1
                max_count = -1
                
                for c_id, counts in cluster_skin_counts.items():
                    if counts[target] > max_count:
                        max_count = counts[target]
                        best_cluster_id = c_id
                
                if best_cluster_id != -1:
                    cluster_profile[best_cluster_id]['dominant_skin_types'].append(target)
                    print(f"   -> Assigned '{target}' to Cluster {best_cluster_id}")

        final_accuracy = float(accuracy_score(actual_labels_list, predicted_labels_list)) if actual_labels_list else 0.0
        final_f1_score = float(f1_score(actual_labels_list, predicted_labels_list, average='weighted')) if actual_labels_list else 0.0

        print(f"[CLUSTERING] Stats - Accuracy: {final_accuracy:.2f}, F1: {final_f1_score:.2f}")

        return dataframe_result, cluster_profile, final_silhouette_value, sum_of_squared_errors, final_accuracy, final_f1_score, clustering_plot_buffer
    
    def generate_cluster_plot(self, pca_features, labels):
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x=pca_features[:, 0], 
            y=pca_features[:, 1], 
            hue=labels, 
            palette='tab10', 
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