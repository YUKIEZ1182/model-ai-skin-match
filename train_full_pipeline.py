import pandas as pd
import numpy as np
import pickle
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter

# Import ฟังก์ชันเดิมจากไฟล์ที่คุณมี
from clean_cosmetic_ingredients import clean_ingredients
from model_training import train_from_scratch

# ตั้งค่าโฟลเดอร์
DATA_DIR = "data"
OUTPUT_DIR = "model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_combine_data(data_dir):
    """อ่านไฟล์ CSV ทั้งหมดในโฟลเดอร์ data มารวมกัน"""
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    # กรองเอาเฉพาะไฟล์ที่เป็น Raw Data (ไม่เอาไฟล์ที่ clean แล้ว)
    raw_files = [f for f in all_files if "cleaned" not in f and "report" not in f]
    
    print(f"📂 Found {len(raw_files)} data files: {[os.path.basename(f) for f in raw_files]}")
    
    df_list = []
    for filename in raw_files:
        try:
            temp_df = pd.read_csv(filename)
            # เพิ่มคอลัมน์ skin_type ถ้ายัังไม่มี (สำหรับไฟล์เก่า)
            if 'skin_type' not in temp_df.columns:
                temp_df['skin_type'] = np.nan
            df_list.append(temp_df)
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")
            
    if not df_list:
        raise ValueError("No CSV files found in data directory!")
        
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"✅ Combined Data: {len(combined_df)} total rows")
    return combined_df

def process_skin_type_column(df):
    """แปลงข้อมูล Skin Type ให้เป็น List และจัดการค่าว่าง"""
    def clean_skin_type(x):
        if pd.isna(x) or str(x).strip() == "":
            return []
        return [s.strip().lower() for s in str(x).split(',')]
    
    return df['skin_type'].apply(clean_skin_type)

def main():
    # 1. Load & Combine Data
    df = load_and_combine_data(DATA_DIR)
    
    # 2. Clean Ingredients
    print("Cleaning ingredients...")
    # แปลงเป็น string ก่อนกัน error
    df['ingredients'] = df['ingredients'].astype(str)
    df['clean_ingredients'] = df['ingredients'].apply(clean_ingredients)
    
    # ลบแถวที่ไม่มีส่วนผสม (หลังจาก clean)
    df = df[df['clean_ingredients'] != ""]
    
    # ==========================================
    # PART 1: Update Association Model
    # ==========================================
    print("\n--- Training Association Rules ---")
    # ใช้ train_from_scratch สร้างโมเดลใหม่จากข้อมูลรวม
    train_from_scratch(df, cleaned_csv=os.path.join(DATA_DIR, "cosmetics_cleaned_final.csv"))
    print("Association Model Updated!")

    # ==========================================
    # PART 2: New K-Means with Skin Type Logic
    # ==========================================
    print("\n--- Training K-Means & Skin Type Mapping ---")
    
    # 2.1 Vectorize
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(', '), token_pattern=None)
    X = vectorizer.fit_transform(df['clean_ingredients'])
    
    # 2.2 Clustering
    K = 5 # ปรับจำนวนกลุ่มได้ตามต้องการ
    kmeans = KMeans(n_clusters=K, random_state=42)
    clusters = kmeans.fit_predict(X)
    df['cluster'] = clusters
    
    # 2.3 Analyze Skin Type per Cluster
    df['skin_type_list'] = process_skin_type_column(df)
    
    cluster_profile = {}
    feature_names = np.array(vectorizer.get_feature_names_out())

    for i in range(K):
        cluster_products = df[df['cluster'] == i]
        
        # นับ Skin Type เฉพาะจากสินค้าที่มีข้อมูล (ไฟล์ใหม่)
        all_skin_types = [st for sublist in cluster_products['skin_type_list'] for st in sublist]
        
        if all_skin_types:
            skin_counts = Counter(all_skin_types)
            top_skin_types = [st for st, count in skin_counts.most_common(3)]
        else:
            top_skin_types = ["unknown"] # กรณีกลุ่มนี้มาจากไฟล์เก่าล้วนๆ
            
        # หา Key Ingredients (Centroid)
        centroid = kmeans.cluster_centers_[i]
        top_indices = centroid.argsort()[-10:][::-1]
        key_ingredients = feature_names[top_indices].tolist()
        
        cluster_profile[i] = {
            "dominant_skin_types": top_skin_types,
            "key_ingredients": key_ingredients,
            "product_count": len(cluster_products)
        }
        
        print(f"Cluster {i} ({len(cluster_products)} products):")
        print(f"  - Suitable for: {top_skin_types}")
        print(f"  - Key Ingredients: {key_ingredients[:3]}...")

    # 2.4 Save Models
    print(f"\nSaving K-Means models to {OUTPUT_DIR}...")
    with open(os.path.join(OUTPUT_DIR, "cluster_profile.pkl"), "wb") as f:
        pickle.dump(cluster_profile, f)
        
    # (Optional: ถ้าจะ save vectorizer/kmeans model ด้วยก็เพิ่มตรงนี้)
        
    print("All Done! Ready to use.")

if __name__ == "__main__":
    main()