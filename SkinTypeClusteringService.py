import pandas as pd
import numpy as np
import pickle
import os
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter

class SkinTypeClusteringService:
    def __init__(self):
        self.ingredient = []
        self.MODEL_DIR = "model"
        os.makedirs(self.MODEL_DIR, exist_ok=True)

    def findClusterForSkinType(self, df):
        vectorizer = TfidfVectorizer(max_df=0.7, min_df=2, stop_words=['water', 'glycerin', 'aqua']) 
        X = vectorizer.fit_transform(df['clean_ingredients'].fillna(''))

        n_samples = X.shape[0]
        if n_samples < 3:
            k = 1
        else:
            actual_max_k = min(10, n_samples - 1)
            best_k, best_score = 2, -1
            
            for test_k in range(2, actual_max_k + 1):
                km = KMeans(n_clusters=test_k, n_init=10, random_state=42)
                labels = km.fit_predict(X)
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = test_k
            k = best_k

        # Run final KMeans
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42) if k > 1 else None
        
        df_result = df.copy()
        df_result['cluster'] = kmeans.fit_predict(X) if kmeans else 0

        cluster_profile = {}
        feature_names = np.array(vectorizer.get_feature_names_out())

        for i in range(k):
            cluster_products = df_result[df_result['cluster'] == i]
            all_skin_types = []
            
            for st in cluster_products['skin_type'].dropna():
                try:
                    st_str = str(st)
                    if '[' in st_str: 
                        all_skin_types.extend(ast.literal_eval(st_str))
                    else: 
                        all_skin_types.extend([s.strip() for s in st_str.split(',') if s.strip()])
                except: 
                    continue
            
            top_types = [st for st, count in Counter(all_skin_types).most_common(2)]
            
            if kmeans:
                centroid = kmeans.cluster_centers_[i]
                top_indices = centroid.argsort()[-10:][::-1]
                key_ings = feature_names[top_indices].tolist()
            else:
                key_ings = feature_names[X.toarray().mean(axis=0).argsort()[-10:][::-1]].tolist()
            
            cluster_profile[i] = {
                "dominant_skin_types": top_types if top_types else ["all skin types"],
                "key_ingredients": key_ings
            }
        
        self.ingredient = df['clean_ingredients'].tolist()

        profile_path = os.path.join(self.MODEL_DIR, "cluster_profile.pkl")
        with open(profile_path, "wb") as f:
            pickle.dump(cluster_profile, f)
            
        return df_result, cluster_profile