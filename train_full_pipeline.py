import pandas as pd
import numpy as np
import pickle
import os
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter

# Import ฟังก์ชันจากไฟล์ที่มีอยู่เดิม
from clean_cosmetic_ingredients import clean_ingredients
from model_training import train_from_scratch

# ตั้งค่าเส้นทางไฟล์
DATA_DIR = "data"
OUTPUT_DIR = "model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_optimal_k(X, max_k=10):
    """
    หลักการ: Silhouette Method
    หาค่า K (จำนวนกลุ่ม) ที่ดีที่สุดโดยวัดความใกล้ชิดของข้อมูลภายในกลุ่ม 
    และความห่างระหว่างกลุ่ม (ค่าที่เข้าใกล้ 1 คือดีที่สุด)
    """
    n_samples = X.shape[0]
    if n_samples < 2:
        return 1
        
    actual_max_k = min(max_k, n_samples - 1)
    best_k = 2
    best_score = -1
    
    print(f"🔍 Finding optimal K (range 2 to {actual_max_k})...")
    
    for k in range(2, actual_max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"✅ Optimal K found: {best_k} (Silhouette Score: {best_score:.4f})")
    return best_k

def run_pipeline():
    # 1. Load Data
    cleaned_csv = os.path.join(DATA_DIR, "cosmetics_cleaned_final.csv")
    if not os.path.exists(cleaned_csv):
        print(f"❌ Error: {cleaned_csv} not found.")
        return

    df = pd.read_csv(cleaned_csv)
    print(f"📂 Loaded {len(df)} products for training.")

    # 2. Vectorization (TF-IDF)
    # หลักการ: TF-IDF ช่วยลดความสำคัญของคำที่พบบ่อยเกินไป (เช่น Water) ด้วย max_df=0.8
    # และตัดส่วนผสมที่พบน้อยเกินไป (Noise) ด้วย min_df=2
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=2) 
    X = vectorizer.fit_transform(df['clean_ingredients'].fillna(''))

    # 3. K-Means with Statistical Optimal K
    k = find_optimal_k(X)
    if k > 1:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        df['cluster'] = kmeans.fit_predict(X)
    else:
        df['cluster'] = 0
        print("⚠️ Not enough data for clustering. Assigned all to cluster 0.")

    # 4. Generate Cluster Profiles
    cluster_profile = {}
    feature_names = np.array(vectorizer.get_feature_names_out())

    for i in range(k if k > 1 else 1):
        cluster_products = df[df['cluster'] == i]
        
        # จัดการข้อมูล Skin Type (แปลงจาก String เป็น List)
        all_skin_types = []
        for st in cluster_products['skin_type'].dropna():
            try:
                # รองรับ format ['oily', 'dry'] หรือ oily, dry
                st_str = str(st)
                if '[' in st_str:
                    all_skin_types.extend(ast.literal_eval(st_str))
                else:
                    all_skin_types.extend([s.strip() for s in st_str.split(',') if s.strip()])
            except:
                continue
        
        # เลือกสภาพผิวที่โดดเด่นที่สุดในกลุ่มนั้น 2 อันดับแรก
        top_skin_types = [st for st, count in Counter(all_skin_types).most_common(2)]
        
        # หา Key Ingredients (ส่วนผสมหลักที่เป็นตัวแทนของกลุ่ม) จาก Centroid
        if k > 1:
            centroid = kmeans.cluster_centers_[i]
            top_indices = centroid.argsort()[-10:][::-1]
            key_ingredients = feature_names[top_indices].tolist()
        else:
            key_ingredients = feature_names[X.toarray().mean(axis=0).argsort()[-10:][::-1]].tolist()
        
        cluster_profile[i] = {
            "dominant_skin_types": top_skin_types if top_skin_types else ["all skin types"],
            "key_ingredients": key_ingredients,
            "product_count": len(cluster_products)
        }

    # 5. Save Model Output
    profile_path = os.path.join(OUTPUT_DIR, "cluster_profile.pkl")
    with open(profile_path, "wb") as f:
        pickle.dump(cluster_profile, f)
    print(f"💾 Cluster profile saved to {profile_path}")

    # 6. Train Association Rules ต่อ
    # เรียกใช้ฟังก์ชันเดิมเพื่อหาความสัมพันธ์ของส่วนผสม (Association Rules)
    # โดยใช้ข้อมูลล่าสุดที่มีการจัดการ Entity Resolution เรียบร้อยแล้ว
    print("🔄 Training association rules...")
    train_from_scratch(df, cleaned_csv=cleaned_csv)
    
    print("🚀 Full pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()