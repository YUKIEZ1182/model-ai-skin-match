from flask import Flask, jsonify, request
import pandas as pd
import pickle
import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from ReTrainService import ReTrainService

load_dotenv()
app = Flask(__name__)

DIRECTUS_URL = os.getenv("DIRECTUS_URL")
TOKEN = os.getenv("DIRECTUS_STATIC_TOKEN")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

AI_LIFT_THRESHOLD = float(os.getenv("AI_LIFT", "1.0"))
AI_CONFIDENCE_THRESHOLD = float(os.getenv("AI_CONFIDENCE", "0.5"))

# Global variables for models in RAM
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

        # Loading Clustering Model
        clustering_response = requests.get(f"{DIRECTUS_URL}/assets/{clustering_file_id}", headers=HEADERS)
        clustering_model_memory = pickle.loads(clustering_response.content)
        
        # Loading Association Rules
        association_response = requests.get(f"{DIRECTUS_URL}/assets/{association_file_id}", headers=HEADERS)
        association_rules_memory = pickle.loads(association_response.content)
        
        print(f"[SERVER] Resources loaded successfully. Log ID: {latest_log['id']}")
    except Exception as error:
        print(f"[SERVER] Error loading resources: {error}")

load_resources()

# --- 1. Related Ingredients Recommendation (With Match Strength) ---
@app.route('/related-ingredients-recommend', methods=['GET'])
def get_recommend_by_ingredient():
    if association_rules_memory is None: 
        return jsonify({"error": "Model not initialized"}), 500
    
    ingredient_parameter = request.args.get('ingredient', '').lower()
    
    filtered_rules = association_rules_memory[
        (association_rules_memory['lift'] >= AI_LIFT_THRESHOLD) & 
        (association_rules_memory['confidence'] >= AI_CONFIDENCE_THRESHOLD)
    ].copy()
    
    if ingredient_parameter:
        query_list = [item.strip() for item in ingredient_parameter.split(',') if item.strip()]
        filtered_rules = filtered_rules[filtered_rules['antecedents'].apply(
            lambda x: any(query in [item.lower() for item in x] for query in query_list)
        )]
    
    # Calculate match_strength for the output (Normalization against Lift 5.0)
    results = filtered_rules.head(10).to_dict(orient='records')
    for row in results:
        strength = min(100, round((row['lift'] / 5.0) * 100)) 
        row['match_strength'] = f"{strength}%"
    
    return jsonify(results)

# --- 2. Skin Type Recommendation (Smart Aggregation Logic) ---
@app.route('/skin-type-recommend', methods=['GET'])
def get_recommend_by_skin_type():
    if clustering_model_memory is None: 
        return jsonify({"error": "Model not initialized"}), 500
    
    skin_type = request.args.get('skinType', '').lower().strip()
    if not skin_type:
        return jsonify({"error": "Missing skinType parameter"}), 400

    # Retrieve profiles containing pre-aggregated clusters from AI Service
    profiles = clustering_model_memory.get('skin_type_profiles', {})
    data = profiles.get(skin_type)

    if data:
        target_clusters = data['target_clusters']
        
        # Note: To return actual products, you would typically filter your product 
        # database using the 'target_clusters' IDs.
        return jsonify({
            "skinType": skin_type, 
            "targetClusters": target_clusters,
            "recommendedIngredients": data['key_ingredients'][:10],
            "message": f"Successfully fetched recommendations from {len(target_clusters)} clusters."
        })
            
    return jsonify({"error": f"Profile for '{skin_type}' not found"}), 404

# --- 3. Control System (Retrain & Reload) ---
@app.route('/retrain', methods=['POST'])
def trigger_retraining_process():
    try:
        print("[SERVER] Retraining process triggered...")  
        retraining_service = ReTrainService(DIRECTUS_URL, TOKEN) 
        retraining_service.triggerRetraining()
        
        load_resources() # Reload updated models into RAM
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
        load_resources() 
        return jsonify({
            "status": "success", 
            "message": "AI Models have been reloaded successfully."
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)