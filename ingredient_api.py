from flask import Flask, jsonify, request
import pandas as pd
import io
import pickle
import os
from datetime import datetime

from train_full_pipeline import run_pipeline
from clean_cosmetic_ingredients import clean_ingredients
from model_training import update_model 

app = Flask(__name__)

# --- ฟังก์ชันช่วย (Helper Functions) ---

def load_rules_model(model_file=os.path.join("model", "association_model.pkl")):
    """โหลดโมเดล Association Rules จากไฟล์ Pickle"""
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            rules_df = pickle.load(f)
        return rules_df.sort_values(by=["lift", "confidence", "support"], ascending=False)
    else:
        raise FileNotFoundError("Association model file not found. Please run training first.")

# --- API Endpoints ---

@app.route('/rules', methods=['GET'])
def get_rules():
    """ดึงข้อมูลความสัมพันธ์ของส่วนผสม (Association Rules) พร้อมการกรอง"""
    ingredient_param = request.args.get('ingredient', None)
    
    try:
        rules_df = load_rules_model()
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    # กรองเฉพาะกฎที่มีนัยสำคัญ (Lift > 1 และ Confidence > 0.7)
    filtered_df = rules_df[(rules_df['lift'] > 1) & (rules_df['confidence'] > 0.7)]
    
    if ingredient_param:
        query_ingredients = [i.strip().lower() for i in ingredient_param.split(',') if i.strip()]
        
        def rule_matches(row):
            union_ing = [ing.lower() for ing in (row['antecedents'] + row['consequents'])]
            return any(q in union_ing for q in query_ingredients)
        
        filtered_df = filtered_df[filtered_df.apply(rule_matches, axis=1)]
        
        def contains_all(row):
            union_ing = [ing.lower() for ing in (row['antecedents'] + row['consequents'])]
            return all(q in union_ing for q in query_ingredients)
        
        filtered_df = filtered_df.copy()
        filtered_df['match_all'] = filtered_df.apply(lambda row: int(contains_all(row)), axis=1)
        filtered_df = filtered_df.sort_values(by=["match_all", "lift", "confidence", "support"], ascending=False)
    else:
        filtered_df = filtered_df.sort_values(by=["lift", "confidence", "support"], ascending=False)
    
    result = []
    for _, row in filtered_df.iterrows():
        result.append({
            "antecedents": row['antecedents'],
            "consequents": row['consequents'],
            "support": row['support'],
            "confidence": row['confidence'],
            "lift": row['lift']
        })
    
    return jsonify(result)

@app.route('/skin-type-recommend', methods=['GET'])
def recommend_products():
    """แนะนำส่วนผสมที่เหมาะสมตามสภาพผิว โดยใช้ผลลัพธ์จาก K-Means Clustering"""
    user_skin_type = request.args.get('skin_type', '').lower().strip()
    profile_path = os.path.join("model_output", "cluster_profile.pkl")
    
    if not os.path.exists(profile_path):
        return jsonify({"error": "Model profile not found. Please run retrain first."}), 500
        
    try:
        with open(profile_path, "rb") as f:
            cluster_profile = pickle.load(f)
    except Exception as e:
        return jsonify({"error": f"Error loading model profile: {str(e)}"}), 500
        
    target_ingredients = []
    
    # Logic 1: หาแบบตรงตัว (Exact Match)
    for cid, data in cluster_profile.items():
        if any(user_skin_type == st.lower() for st in data['dominant_skin_types']):
            target_ingredients = data['key_ingredients']
            break
            
    # Logic 2: หาแบบบางส่วน (Contains Match)
    if not target_ingredients:
        for cid, data in cluster_profile.items():
            for st in data['dominant_skin_types']:
                if user_skin_type in st.lower() or st.lower() in user_skin_type:
                    target_ingredients = data['key_ingredients']
                    break
        
    if not target_ingredients:
        return jsonify({"error": f"No recommendation found for skin type: {user_skin_type}"}), 404

    # ส่งคืน Top 5 ส่วนผสมเพื่อลด Noise
    return jsonify({
        "skin_type": user_skin_type,
        "recommended_ingredients": target_ingredients[:5] 
    })

@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Endpoint สำหรับรับข้อมูลชุดใหญ่จาก Directus เพื่อ Update ข้อมูลใน CSV และสั่ง Retrain
    ใช้หลักการ:
    1. Entity Integrity: ตรวจสอบ ID เป็นหลัก ถ้าเจอให้ Update
    2. Entity Resolution: ถ้า ID ไม่เจอแต่ชื่อซ้ำ ให้ถือเป็นสินค้าเดิมและ Update ID
    3. Duplicate Handling: สินค้าต่างแบรนด์ที่ส่วนผสมเหมือนกันจะถูกเก็บแยกแถว (เพราะ ID ต่างกัน)
    """
    new_items = request.json
    if not new_items or not isinstance(new_items, list):
        return jsonify({"error": "Invalid data format. Expected a list."}), 400

    cleaned_csv = os.path.join("data", "cosmetics_cleaned_final.csv")
    
    try:
        df = pd.read_csv(cleaned_csv)
    except FileNotFoundError:
        # กรณีรันครั้งแรกแล้วไม่มีไฟล์เดิม
        df = pd.DataFrame(columns=['directus_id', 'product_name', 'ingredients', 'clean_ingredients', 'skin_type'])

    # ตรวจสอบว่ามีคอลัมน์ directus_id หรือยัง
    if 'directus_id' not in df.columns:
        df['directus_id'] = None

    updated_count = 0
    new_count = 0

    for item in new_items:
        d_id = item.get('id')
        name = item.get('name', '').strip()
        ing_raw = item.get('ingredients', '').strip()
        skin_types = item.get('skin_types', [])
        
        clean_ing = clean_ingredients(ing_raw)
        
        # 1. ตรวจสอบจาก ID (Entity Integrity)
        mask_id = df['directus_id'] == d_id
        if mask_id.any():
            # เจอ ID เดิม -> อัปเดตข้อมูลทับ (รองรับเปลี่ยนชื่อแบรนด์หรือเปลี่ยนสูตร)
            df.loc[mask_id, ['product_name', 'ingredients', 'clean_ingredients', 'skin_type']] = \
                [name, ing_raw, clean_ing, str(skin_types)]
            updated_count += 1
        else:
            # 2. ไม่เจอ ID แต่ "ชื่อสินค้า" ซ้ำ (Entity Resolution)
            mask_name = df['product_name'].str.lower() == name.lower()
            if mask_name.any():
                # อัปเดต ID และข้อมูลอื่นๆ ลงในสินค้าชื่อเดิมที่มีอยู่
                df.loc[mask_name, ['directus_id', 'ingredients', 'clean_ingredients', 'skin_type']] = \
                    [d_id, ing_raw, clean_ing, str(skin_types)]
                updated_count += 1
            else:
                # 3. สินค้าใหม่ (ไม่ซ้ำทั้ง ID และ ชื่อ) -> เพิ่มเป็นแถวใหม่
                new_row = {
                    "directus_id": d_id,
                    "product_name": name,
                    "ingredients": ing_raw,
                    "clean_ingredients": clean_ing,
                    "skin_type": str(skin_types)
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                new_count += 1

    # บันทึกข้อมูลลง CSV
    df.to_csv(cleaned_csv, index=False)
    
    try:
        # สั่งรัน Pipeline เพื่อหาค่า K ที่ดีที่สุด และสร้างโมเดลใหม่ทันที
        run_pipeline()
        return jsonify({
            "status": "success",
            "message": "Model retrained successfully.",
            "updated": updated_count,
            "new_added": new_count
        }), 200
    except Exception as e:
        return jsonify({"error": f"Retrain logic failed: {str(e)}"}), 500

# Endpoint เดิมสำหรับการอัปโหลดไฟล์ผ่าน Form-data
@app.route('/upload_data', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    try:
        new_data = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))
        combined_data, rules = update_model(new_data)
        return jsonify({"message": "Data processed", "num_rules": len(rules)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # รันบน Host 0.0.0.0 เพื่อให้เข้าถึงได้จากภายนอก (เช่น Docker/Coolify)
    app.run(host='0.0.0.0', port=5000)