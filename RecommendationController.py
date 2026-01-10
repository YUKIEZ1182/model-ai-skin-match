from flask import Flask, jsonify, request
import pandas as pd
import pickle
import os
import requests
from io import BytesIO
from dotenv import load_dotenv

from ReTrainService import ReTrainService

load_dotenv()
app = Flask(__name__)

# --- Configuration ---
IS_DEVELOP = os.getenv("DEVELOP_MODE", "false").lower() == "true"
DIRECTUS_URL = os.getenv("DIRECTUS_URL")
TOKEN = os.getenv("DIRECTUS_STATIC_TOKEN")
CSV_ID = os.getenv("DATASET_FILE_ID")
CLUSTER_ID = os.getenv("CLUSTERING_MODEL_ID")
ASSOC_ID = os.getenv("ASSOCIATION_MODEL_ID")
headers = {"Authorization": f"Bearer {TOKEN}"}

# Global memory storage for models
cluster_mem, rules_mem = None, None

def load_resources():
    """Load models from Directus Assets into RAM for prediction"""
    global cluster_mem, rules_mem
    try:
        if CLUSTER_ID:
            res = requests.get(f"{DIRECTUS_URL}/assets/{CLUSTER_ID}", headers=headers)
            if res.status_code == 200: cluster_mem = pickle.loads(res.content)
        if ASSOC_ID:
            res = requests.get(f"{DIRECTUS_URL}/assets/{ASSOC_ID}", headers=headers)
            if res.status_code == 200: rules_mem = pickle.loads(res.content)
        print("Models synchronized and loaded into RAM.")
    except Exception as e:
        print(f"⚠️ Resource Load Error: {e}")

@app.route('/rules', methods=['GET'])
def getRecommendWithSimilarIngredient():
    """
    Method as defined in Class Diagram: + getRecommendWithSimilarIngredient(ingredient)
   
    """
    if rules_mem is None: 
        return jsonify({"error": "Model not initialized"}), 500
    
    ingredient_param = request.args.get('ingredient', '').lower()
    
    filtered = rules_mem[rules_mem['lift'] > 1.2]
    if ingredient_param:
        query = [i.strip() for i in ingredient_param.split(',') if i.strip()]
        filtered = filtered[filtered['antecedents'].apply(
            lambda x: any(q in [i.lower() for i in x] for q in query)
        )]
    
    return jsonify(filtered.head(10).to_dict(orient='records'))

@app.route('/skin-type-recommend', methods=['GET'])
def getRecommendSuitForSkinType():
    """
    Method as defined in Class Diagram: + getRecommendSuitForSkinType(skinType)
   
    """
    if cluster_mem is None: 
        return jsonify({"error": "Model not initialized"}), 500
    
    skin_type = request.args.get('skinType', '').lower().strip()
    
    for cid, data in cluster_mem.items():
        if any(skin_type in st.lower() for st in data['dominant_skin_types']):
            return jsonify({
                "skinType": skin_type, 
                "recommendedIngredients": data['key_ingredients'][:5]
            })
            
    return jsonify({"error": "Skin type profile not found"}), 404

@app.route('/retrain', methods=['POST'])
def triggerRetraining():
    """
    Maps to ReTrainService.triggerRetraining() as per Diagram relationships.
   
    """
    try:
        # Instantiate the retraining service
        retrain_service = ReTrainService(DIRECTUS_URL, TOKEN, CSV_ID)
        
        # Execute the retraining logic
        df_final, cluster_p, assoc_r = retrain_service.triggerRetraining()
        
        # Helper to sync results back to Directus Assets
        def patch_asset(fid, filename, content, content_type):
            if not fid: return
            buffer = BytesIO()
            if filename.endswith('.pkl'):
                pickle.dump(content, buffer)
            else:
                content.to_csv(buffer, index=False)
            buffer.seek(0)
            requests.patch(
                f"{DIRECTUS_URL}/files/{fid}", 
                headers=headers, 
                files={'data': (filename, buffer, content_type)}
            )

        # Upload updated files
        patch_asset(CSV_ID, 'cosmetics_cleaned_final.csv', df_final, 'text/csv')
        patch_asset(CLUSTER_ID, 'cluster_profile.pkl', cluster_p, 'application/octet-stream')
        patch_asset(ASSOC_ID, 'association_model.pkl', assoc_r, 'application/octet-stream')
        
        # Refresh RAM models
        load_resources()
        
        return jsonify({"status": "success", "message": "Models retrained and synchronized"})
    except Exception as e:
        print(f"Retrain Route Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_resources()
    app.run(host='0.0.0.0', port=5000)