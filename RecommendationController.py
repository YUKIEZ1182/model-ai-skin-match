from flask import Flask, jsonify, request
import pandas as pd
import pickle
import os
import requests
from dotenv import load_dotenv
from ReTrainService import ReTrainService

load_dotenv()
app = Flask(__name__)

DIRECTUS_URL = os.getenv("DIRECTUS_URL")
TOKEN = os.getenv("DIRECTUS_STATIC_TOKEN")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

AI_LIFT_THRESHOLD = float(os.getenv("AI_LIFT", "1.0"))
AI_CONFIDENCE_THRESHOLD = float(os.getenv("AI_CONFIDENCE", "0.5"))

clustering_model_memory = None
association_rules_memory = None

def load_resources():
    global clustering_model_memory, association_rules_memory
    try:
        print("[SERVER] Synchronizing active models from Directus...")
        
        log_query = "/items/model_training_log?sort=-date_trained&filter[is_active][_eq]=true&limit=1"
        log_response = requests.get(f"{DIRECTUS_URL}{log_query}", headers=HEADERS).json()
        
        if not log_response.get('data'):
            print("[SERVER] Warning: No active model found in logs.")
            return

        latest_log = log_response['data'][0]
        clustering_file_id = latest_log['clustering_model_file']
        association_file_id = latest_log['association_model_file']

        clustering_response = requests.get(f"{DIRECTUS_URL}/assets/{clustering_file_id}", headers=HEADERS)
        clustering_model_memory = pickle.loads(clustering_response.content)
        
        association_response = requests.get(f"{DIRECTUS_URL}/assets/{association_file_id}", headers=HEADERS)
        association_rules_memory = pickle.loads(association_response.content)
        
        print(f"[SERVER] Resources loaded successfully (Log ID: {latest_log['id']})")
    except Exception as error:
        print(f"[SERVER] Error loading resources: {error}")

load_resources()

@app.route('/related-ingredients-recommend', methods=['GET'])
def get_recommend_by_ingredient():
    if association_rules_memory is None: 
        return jsonify({"error": "Model not initialized"}), 500
    
    ingredient_parameter = request.args.get('ingredient', '').lower()
    
    filtered_rules = association_rules_memory[
        (association_rules_memory['lift'] >= AI_LIFT_THRESHOLD) & 
        (association_rules_memory['confidence'] >= AI_CONFIDENCE_THRESHOLD)
    ]
    
    if ingredient_parameter:
        query_list = [item.strip() for item in ingredient_parameter.split(',') if item.strip()]
        filtered_rules = filtered_rules[filtered_rules['antecedents'].apply(
            lambda x: any(query in [item.lower() for item in x] for query in query_list)
        )]
    
    return jsonify(filtered_rules.head(10).to_dict(orient='records'))

@app.route('/skin-type-recommend', methods=['GET'])
def get_recommend_by_skin_type():
    if clustering_model_memory is None: 
        return jsonify({"error": "Model not initialized"}), 500
    
    skin_type = request.args.get('skinType', '').lower().strip()
    if not skin_type:
        return jsonify({"error": "Missing skinType parameter"}), 400

    for cluster_id, data in clustering_model_memory.items():
        if any(skin_type in type_label.lower() for type_label in data['dominant_skin_types']):
            return jsonify({
                "skinType": skin_type, 
                "recommendedIngredients": data['key_ingredients'][:5]
            })
            
    return jsonify({"error": f"Profile for '{skin_type}' not found"}), 404

@app.route('/retrain', methods=['POST'])
def trigger_retraining_process():
    try:
        print("[SERVER] Retraining process triggered...")  
        retraining_service = ReTrainService(DIRECTUS_URL, TOKEN) 
        retraining_service.triggerRetraining()
        
        load_resources()
        return jsonify({
            "status": "success", 
            "message": "Retraining completed and resources updated."
        }), 200
    except Exception as error:
        print(f"[SERVER] Retraining process failed: {error}")
        return jsonify({"error": str(error)}), 500

@app.route('/reload', methods=['GET'])
def reload_models():
    try:
        # เรียกฟังก์ชันเดิมที่เราใช้ตอน start server
        load_resources() 
        return jsonify({
            "status": "success", 
            "message": "AI Models have been reloaded into RAM successfully"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)