import pandas as pd
import numpy as np
import requests
import pickle
import io
import re
from datetime import datetime, timedelta, timezone 
from io import BytesIO
from IngredientAssociationService import IngredientAssociationService
from SkinTypeClusteringService import SkinTypeClusteringService

class ReTrainService:
    def __init__(self, directus_url, token, csv_id=None):
        self.directus_url = directus_url
        self.headers = {"Authorization": f"Bearer {token}"}
        self.association_service = IngredientAssociationService()
        self.clustering_service = SkinTypeClusteringService()
        
        self.ingredients_ignore_list = {
            'water', 'aqua', 'eau', 'glycerin', 'phenoxyethanol', 'butylene glycol',
            'ethylhexylglycerin', 'sodium hyaluronate', 'disodium edta', 'tocopherol'
        }

    def _cleanIngredients(self, ingredients_string):
        if pd.isna(ingredients_string) or str(ingredients_string).strip() == "":
            return ""
        
        processed_string = str(ingredients_string).replace("/", ",").replace(";", ",").replace(" and ", ",")
        tokens = processed_string.split(",")
        cleaned_ingredients_list = [
            token.strip().lower() for token in tokens 
            if len(token.strip()) > 2 and token.strip().lower() not in self.ingredients_ignore_list
        ]
        return ", ".join(list(dict.fromkeys(cleaned_ingredients_list)))

    def triggerRetraining(self):
        try:
            print("\n" + "="*50)
            print("[RETRAIN] Starting model training pipeline")
            
            now_utc = datetime.now(timezone.utc)
            timestamp = now_utc.strftime("%Y%m%d_%H%M%S")
            current_training_time = now_utc.isoformat().replace("+00:00", "Z")
            
            base_dataframe = pd.DataFrame()
            last_training_time = None
            
            log_query = "/items/model_training_log?sort=-date_trained&filter[is_active][_eq]=true&limit=1"
            log_response = requests.get(f"{self.directus_url}{log_query}", headers=self.headers).json()
            
            if log_response.get('data') and len(log_response['data']) > 0:
                latest_log = log_response['data'][0]
                last_training_time = latest_log.get('date_trained')
                csv_file_id = latest_log.get('dataset_file')
                
                if csv_file_id:
                    csv_response = requests.get(f"{self.directus_url}/assets/{csv_file_id}", headers=self.headers)
                    if csv_response.status_code == 200:
                        base_dataframe = pd.read_csv(BytesIO(csv_response.content))

            if base_dataframe.empty:
                base_dataframe = pd.read_csv('cosmetics_final.csv')
            
            if 'Label' in base_dataframe.columns:
                base_dataframe = base_dataframe.rename(columns={'Label': 'skin_type'})
            if 'product_name' in base_dataframe.columns:
                base_dataframe = base_dataframe.rename(columns={'product_name': 'name'})

            database_url = f"{self.directus_url}/items/product?fields=id,name,suitable_skin_type,ingredients.ingredient_id.name,date_created,date_updated&limit=-1"
            
            if last_training_time:
                last_date_time = datetime.fromisoformat(last_training_time.replace('Z', '+00:00'))
                buffer_string = (last_date_time - timedelta(days=1)).isoformat().split('+')[0]
                database_url += f"&filter[_or][0][date_created][_gt]={buffer_string}&filter[_or][1][date_updated][_gt]={buffer_string}"

            new_products_data = requests.get(database_url, headers=self.headers).json().get('data', [])
            new_items_list = []
            for item in new_products_data:
                ingredients_list = [
                    ingredient['ingredient_id']['name'] 
                    for ingredient in item.get('ingredients', []) 
                    if ingredient.get('ingredient_id')
                ]
                if ingredients_list:
                    new_items_list.append({
                        "id": str(item['id']), 
                        "name": item['name'], 
                        "skin_type": str(item.get('suitable_skin_type', [])), 
                        "ingredients": ", ".join(ingredients_list)
                    })
            
            new_products_dataframe = pd.DataFrame(new_items_list)
            total_products_dataframe = pd.concat([base_dataframe, new_products_dataframe], ignore_index=True)
            total_products_dataframe['unique_key'] = total_products_dataframe['id'].fillna(total_products_dataframe['name'])
            total_products_dataframe = total_products_dataframe.drop_duplicates(subset=['unique_key'], keep='last').drop(columns=['unique_key'])

            def clean_label(val):
                if pd.isna(val): return 'unknown'
                s = str(val).lower().strip()
                for char in "[]'\"":
                    s = s.replace(char, "")
                if not s or s in ['nan', 'none', 'null']: return 'unknown'
                primary = s.split(',')[0].strip()
                valid_types = ['oily', 'dry', 'combination', 'sensitive']
                return primary if primary in valid_types else 'unknown'
            
            total_products_dataframe['skin_type'] = total_products_dataframe['skin_type'].apply(clean_label)
            total_products_dataframe['clean_ingredients'] = total_products_dataframe['ingredients'].apply(self._cleanIngredients)
            total_products_dataframe = total_products_dataframe[total_products_dataframe['clean_ingredients'] != ""].reset_index(drop=True)

            (
                labeled_dataframe, 
                cluster_profile, 
                silhouette_score, 
                sum_of_squared_errors, 
                accuracy_score, 
                f1_score, 
                clustering_plot_buffer
            ) = self.clustering_service.findClusterForSkinType(total_products_dataframe)
            
            (
                association_rules, 
                support_value, 
                confidence_value, 
                lift_value
            ) = self.association_service.findRelatedIngredient(labeled_dataframe)
            
            association_plot_buffer = self.association_service.generate_plot(association_rules)

            def upload_file_to_directus(name, content, mime_type):
                upload_response = requests.post(
                    f"{self.directus_url}/files", 
                    headers=self.headers, 
                    files={'data': (name, content, mime_type)}
                )
                return upload_response.json()['data']['id']

            association_rules_csv_content = association_rules.to_csv(index=False).encode('utf-8')
            dataset_csv_content = labeled_dataframe.to_csv(index=False).encode('utf-8')

            association_model_id = upload_file_to_directus(f"association_model_{timestamp}.pkl", pickle.dumps(association_rules), "application/octet-stream")
            clustering_model_id = upload_file_to_directus(f"clustering_model_{timestamp}.pkl", pickle.dumps(cluster_profile), "application/octet-stream")
            association_rules_csv_id = upload_file_to_directus(f"association_rules_{timestamp}.csv", association_rules_csv_content, "text/csv")
            dataset_file_id = upload_file_to_directus(f"dataset_file_{timestamp}.csv", dataset_csv_content, "text/csv")
            clustering_visualization_id = upload_file_to_directus(f"clustering_visualization_{timestamp}.png", clustering_plot_buffer, "image/png")
            association_visualization_id = upload_file_to_directus(f"association_visualization_{timestamp}.png", association_plot_buffer, "image/png")

            training_log_payload = {
                "date_trained": current_training_time,
                "silhouette_score": float(silhouette_score),
                "sum_of_squared_errors": float(sum_of_squared_errors),
                "accuracy": float(accuracy_score),
                "f1_score": float(f1_score),
                "support": float(support_value),
                "confidence": float(confidence_value),
                "lift": float(lift_value),
                "cluster_visualization": clustering_visualization_id,
                "association_visualization": association_visualization_id,
                "dataset_file": dataset_file_id,
                "association_rules_csv": association_rules_csv_id,
                "association_model_file": association_model_id,
                "clustering_model_file": clustering_model_id,
                "is_active": True
            }

            active_logs_response = requests.get(f"{self.directus_url}/items/model_training_log?filter[is_active][_eq]=true", headers=self.headers).json().get('data', [])
            for log in active_logs_response:
                requests.patch(f"{self.directus_url}/items/model_training_log/{log['id']}", headers=self.headers, json={"is_active": False})
            
            requests.post(f"{self.directus_url}/items/model_training_log", headers=self.headers, json=training_log_payload)
            
            print(f"[RETRAIN] Completed. Accuracy: {float(accuracy_score):.4f}, F1: {float(f1_score):.4f}")
            return labeled_dataframe, cluster_profile, association_rules

        except Exception as error:
            print(f"[RETRAIN] Error encountered: {error}")
            raise error