import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from io import BytesIO

class SkinTypeClusteringService:
    def __init__(self):
        self.model_directory = "model"
        os.makedirs(self.model_directory, exist_ok=True)
        
        self.stop_words_list = [
            'water', 'aqua', 'eau', 'glycerin', 'phenoxyethanol', 'butylene', 'glycol', 
            'dimethicone', 'sodium', 'chloride', 'citric', 'acid', 'xanthan', 'gum', 
            'disodium', 'edta', 'benzoate', 'potassium', 'sorbate', 'ethylhexylglycerin',
            'fragrance', 'parfum', 'alcohol', 'denat', 'methylparaben', 'propylparaben',
            'stearate', 'palmitate', 'acetate', 'tocopherol', 'carbomer', 'cetyl', 'stearyl',
            'peg', 'polymer', 'hydrogenated', 'extract', 'leaf', 'root', 'oil'
        ]
        
        self.feature_weights = {
            'centella': 30.0, 'madecassoside': 25.0, 'panthenol': 25.0,
            'niacinamide': 20.0, 'witch hazel': 15.0, 'zinc pca': 15.0,
            'ceramide': 12.0, 'shea': 12.0, 'salicylic': 10.0
        }

    def _prepare_labels(self, dataframe):
        priority = ['sensitive', 'combination', 'dry', 'oily']
        def get_primary(st):
            if pd.isna(st): return 'unknown'
            st = str(st).lower()
            for p in priority:
                if p in st: return p
            return 'unknown'
        df = dataframe.copy()
        df['primary_skin_type'] = df['skin_type'].apply(get_primary)
        return df

    def findClusterForSkinType(self, dataframe):
        print("[AI] Initializing unsupervised clustering and identity extraction...")
        df_processed = self._prepare_labels(dataframe)
        ing_col = 'ingredients' if 'ingredients' in df_processed.columns else 'clean_ingredients'
        
        vectorizer = TfidfVectorizer(max_df=0.12, min_df=2, ngram_range=(1, 2), 
                                     stop_words=self.stop_words_list, sublinear_tf=True)
        tfidf_matrix = vectorizer.fit_transform(df_processed[ing_col].fillna(''))
        tfidf_dense = tfidf_matrix.toarray()
        feature_names = vectorizer.get_feature_names_out()
        
        vocab = vectorizer.vocabulary_
        for word, weight in self.feature_weights.items():
            if word in vocab:
                tfidf_dense[:, vocab[word]] *= weight

        pca = PCA(n_components=8, random_state=42)
        pca_features = pca.fit_transform(tfidf_dense)

        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=150, max_iter=2000, random_state=42)
        df_processed['cluster'] = kmeans.fit_predict(pca_features)
        
        sse_score = kmeans.inertia_
        
        all_skins = ['sensitive', 'combination', 'dry', 'oily']
        cluster_stats = []
        for i in range(1, n_clusters):
            subset = df_processed[df_processed['cluster'] == i]
            total_in_cluster = len(subset)
            counts = subset['primary_skin_type'].value_counts()
            
            for st in all_skins:
                count = counts.get(st, 0)
                density = count / total_in_cluster if total_in_cluster > 0 else 0
                rank = list(counts.index).index(st) + 1 if st in counts.index else 99
                cluster_stats.append({
                    'cluster': i, 'skin_type': st, 'count': count, 'density': density, 'rank': rank
                })

        stats_df = pd.DataFrame(cluster_stats)
        skin_type_profiles = {}

        for st in all_skins:
            st_stats = stats_df[stats_df['skin_type'] == st].copy()
            quality_matches = st_stats[st_stats['rank'] <= 2].sort_values(by=['rank', 'density'], ascending=[True, False])
            
            if not quality_matches.empty:
                final_selection = quality_matches.head(2)['cluster'].tolist()
                mode_desc = "Primary and Secondary Association" # เปลี่ยนคำอธิบายใหม่ให้ดูหรูขึ้น
            else:
                best_single = st_stats.sort_values(by='density', ascending=False).head(1)
                final_selection = best_single['cluster'].tolist()
                mode_desc = "Core Identity Mapping"

            aggregated_ings = []
            for cid in final_selection:
                mean_tfidf = tfidf_dense[df_processed['cluster'] == cid].mean(axis=0)
                top_idx = mean_tfidf.argsort()[-8:][::-1]
                aggregated_ings.extend([feature_names[idx] for idx in top_idx])

            skin_type_profiles[st] = {
                'target_clusters': final_selection,
                'key_ingredients': list(dict.fromkeys(aggregated_ings))[:10],
                'selection_logic': mode_desc
            }

        cluster_analysis = {'skin_type_profiles': skin_type_profiles}
        for i in range(n_clusters):
            subset = df_processed[df_processed['cluster'] == i]
            counts = subset['primary_skin_type'].value_counts()
            p_skin = counts.index[0] if not counts.empty else "unknown"
            
            cluster_analysis[i] = {
                'dominant_skin_types': [p_skin.lower()],
                'primary': p_skin,
                'key_ingredients': [feature_names[idx] for idx in tfidf_dense[df_processed['cluster'] == i].mean(axis=0).argsort()[-10:][::-1]]
            }

        plot_buffer = self.generate_nuanced_plot(df_processed, cluster_analysis, n_clusters)
        sil_score = silhouette_score(pca_features, df_processed['cluster'])
        
        print(f"[AI] Clustering Completed. Silhouette: {sil_score:.3f}, SSE: {sse_score:.2f}")
        return df_processed, cluster_analysis, sil_score, sse_score, plot_buffer

    def generate_nuanced_plot(self, dataframe, analysis, n_clusters):
        plot_data = pd.crosstab(dataframe['cluster'], dataframe['primary_skin_type'])
        target_cols = ['combination', 'oily', 'dry', 'sensitive']
        available_cols = [c for c in target_cols if c in plot_data.columns]
        plot_data = plot_data[available_cols]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), gridspec_kw={'width_ratios': [1.2, 1]})
        colors = {'combination': '#9b59b6', 'oily': '#27ae60', 'dry': '#2980b9', 'sensitive': '#f1c40f'}
        current_colors = [colors[c] for c in available_cols]
        
        plot_data.plot(kind='bar', stacked=True, color=current_colors, alpha=0.8, ax=ax1, edgecolor='white')
        
        # 1. ระบุชื่อแกน X และ Y ให้ชัดเจน
        ax1.set_title(f'Skin Type Cluster Distribution Analysis', fontsize=18)
        ax1.set_xlabel('Cluster Identifier (Index)', fontsize=14)
        ax1.set_ylabel('Total Product Frequency', fontsize=14)
        
        # 2. ย้ายป้าย Skin Type (Legend) ลงมาไว้ด้านล่าง
        ax1.legend(title='Skin Type Profiles', loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                   ncol=4, fontsize=12, frameon=True)

        ax2.axis('off')
        # 3. ปรับชื่อหัวข้อ และลบวงเล็บทั้งหมดออกจากส่วนแสดงผล
        text_content = "AI Profile Selection\n" + "="*55 + "\n\n"
        profiles = analysis['skin_type_profiles']
        for st, data in profiles.items():
            text_content += f"SKIN TYPE: {st.upper()}\n"
            text_content += f"DATA SOURCE: CLUSTERS {data['target_clusters']}\n"
            text_content += f"STRATEGY: {data['selection_logic']}\n" # ไม่มีวงเล็บแล้ว
            text_content += f"INGREDIENTS: {', '.join(data['key_ingredients'][:5])}\n"
            text_content += "-"*55 + "\n"
        
        ax2.text(0, 0.95, text_content, transform=ax2.transAxes, fontsize=11, verticalalignment='top', family='monospace')

        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf