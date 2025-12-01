from flask import Flask, jsonify, request
import pandas as pd
import io
import pickle
import os
from datetime import datetime

from model_training import update_model  # Import update_model from model_training.py

app = Flask(__name__)

def load_rules_model(model_file=os.path.join("model", "association_model.pkl")):
    """
    Load the association rules from the pickled model file.
    """
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            rules_df = pickle.load(f)
        return rules_df.sort_values(by=["lift", "confidence", "support"], ascending=False)
    else:
        raise FileNotFoundError("Association model file not found. Please update the model.")

@app.route('/rules', methods=['GET'])
def get_rules():
    """
    GET /rules returns association rules.
    Supports filtering by ingredients (comma-separated query parameter).
    """
    ingredient_param = request.args.get('ingredient', None)
    
    try:
        rules_df = load_rules_model()
    except Exception as e:
        return jsonify({"error": "Association rules model not found. Please update the model."}), 400
    
    # Filter rules with lift > 1 and confidence > 0.7
    filtered_df = rules_df[(rules_df['lift'] > 1) & (rules_df['confidence'] > 0.7)]
    
    if ingredient_param:
        query_ingredients = [i.strip().lower() for i in ingredient_param.split(',') if i.strip()]
        
        def rule_matches(row):
            # Combine the antecedents and consequents lists
            union_ing = [ing.lower() for ing in (row['antecedents'] + row['consequents'])]
            return any(q in union_ing for q in query_ingredients)
        
        filtered_df = filtered_df[filtered_df.apply(rule_matches, axis=1)]
        
        def contains_all(row):
            union_ing = [ing.lower() for ing in (row['antecedents'] + row['consequents'])]
            return all(q in union_ing for q in query_ingredients)
        
        filtered_df = filtered_df.copy()
        match_all_series = filtered_df.apply(lambda row: int(contains_all(row)), axis=1)
        filtered_df['match_all'] = match_all_series.values.flatten()
        filtered_df = filtered_df.sort_values(by=["match_all", "lift", "confidence", "support"], ascending=False)
    else:
        filtered_df = filtered_df.sort_values(by=["lift", "confidence", "support"], ascending=False)
    
    result = []
    for _, row in filtered_df.iterrows():
        result.append({
            "antecedents": row['antecedents'],  # Already stored as list
            "consequents": row['consequents'],  # Already stored as list
            "support": row['support'],
            "confidence": row['confidence'],
            "lift": row['lift']
        })
    
    return jsonify(result)

@app.route('/upload_data', methods=['POST'])
def upload_data():
    """
    POST /upload_data allows uploading a new CSV file (via form-data key "file")
    to update the model. The updated model is saved to csv_report/ and model/ directories.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading."}), 400

    try:
        new_data = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))
        combined_data, rules = update_model(new_data)
        response = {
            "message": "Data uploaded and model updated successfully.",
            "num_transactions": len(combined_data),
            "num_rules": len(rules)
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"An error occurred during upload: {str(e)}"}), 500

@app.route('/training', methods=['POST'])
def training():
    """
    POST /training?ingredient=<comma-separated ingredients>&product_name=<product name>
    Adds a new transaction to the cleaned data and updates the model.
    If the product name already exists in the cleaned data (data/cosmetics_cleaned.csv),
    no new training is performed and a notification is returned.
    Otherwise, the model is updated and timestamped CSV and pickle files are saved.
    """
    ingredient_param = request.args.get('ingredient', None)
    product_name = request.args.get('product_name', None)
    
    if not ingredient_param:
        return jsonify({"error": "No ingredient provided in query parameters."}), 400
    if not product_name:
        return jsonify({"error": "No product_name provided in query parameters."}), 400

    new_data = pd.DataFrame({
        "product_name": [product_name.lower()],
        "clean_ingredients": [ingredient_param.lower()]
    })
    
    cleaned_csv = os.path.join("data", "cosmetics_cleaned.csv")
    
    if os.path.exists(cleaned_csv):
        df_existing = pd.read_csv(cleaned_csv)
        if "product_name" in df_existing.columns:
            if product_name.lower() in df_existing["product_name"].str.lower().values:
                return jsonify({"message": f"Product '{product_name}' already exists. No update was performed."}), 200

    try:
        combined_data, rules = update_model(new_data, cleaned_csv=cleaned_csv)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rule_filename = f"association_rules_{timestamp}.csv"
        model_filename = f"association_model_{timestamp}.pkl"
        csv_path = os.path.join("csv_report", rule_filename)
        model_path = os.path.join("model", model_filename)
        
        rules.to_csv(csv_path, index=False)
        with open(model_path, "wb") as f:
            pickle.dump(rules, f)
        
        response = {
            "message": "Model updated successfully.",
            "num_transactions": len(combined_data),
            "num_rules": len(rules),
            "rules_file": csv_path,
            "model_file": model_path
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"An error occurred during training: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)