import pandas as pd
import requests
import os
import re
from io import StringIO
from IngredientAssociationService import IngredientAssociationService
from SkinTypeClusteringService import SkinTypeClusteringService

class ReTrainService:
    def __init__(self, directus_url, token, csv_id):
        self.directus_url = directus_url
        self.headers = {"Authorization": f"Bearer {token}"}
        self.csv_id = csv_id
        self.assoc_service = IngredientAssociationService()
        self.cluster_service = SkinTypeClusteringService()
        
        self.IGNORE_LIST = {
            'water', 'aqua', 'eau', 'purified water', 'distilled water',
            'aloe barbadensis leaf water', 'camellia sinensis leaf water', 'rose water',
            'glycerin', 'butylene glycol', 'propylene glycol', 'dipropylene glycol', 
            'propanediol', 'pentylene glycol', 'caprylyl glycol', 'hexylene glycol',
            '1,2-hexanediol', 'peg/ppg-17/6 copolymer', 'glycereth-26', 'sorbitol',
            'alcohol', 'alcohol denat', 'sd alcohol', 'ethanol', 'isopropyl alcohol',
            'fragrance', 'parfum', 'flavor', 'aroma', 
            'limonene', 'linalool', 'geraniol', 'citronellol', 'citral', 'eugenol', 'coumarin',
            'phenoxyethanol', 'ethylhexylglycerin', 'sodium benzoate', 'potassium sorbate',
            'disodium edta', 'tetrasodium edta', 'trisodium edta', 'sodium citrate',
            'citric acid', 'sodium hydroxide', 'potassium hydroxide', 'triethanolamine',
            'chlorphenesin', 'bht', 'tocopherol', 'tocopheryl acetate', 'methylparaben', 'propylparaben',
            'carbomer', 'xanthan gum', 'acrylates/c10-30 alkyl acrylate crosspolymer',
            'dimethicone', 'cyclopentasiloxane', 'cyclohexasiloxane', 'dimethiconol',
            'stearic acid', 'palmitic acid', 'myristic acid', 'lauric acid',
            'cetyl alcohol', 'cetearyl alcohol', 'stearyl alcohol', 'behenyl alcohol',
            'glyceryl stearate', 'peg-100 stearate', 'polysorbate 20', 'polysorbate 60', 'polysorbate 80',
            'hydrogenated lecithin', 'polyacrylate crosspolymer-6', 'ammonium acryloyldimethyltaurate/vp copolymer',
            'titanium dioxide', 'mica', 'tin oxide', 'iron oxides'
        }

    def _cleanIngredients(self, ingredient_str):
        """
        Internal private method to process and clean ingredient strings.
        This hides the cleaning logic inside ReTrainService as part of the data preparation flow.
        """
        if pd.isna(ingredient_str): 
            return ""
        
        # Standardize delimiters
        processed_str = ingredient_str.replace("/", ",").replace(";", ",").replace(" and ", ",")
        tokens = processed_str.split(",")
        cleaned_tokens = []
        
        for token in tokens:
            #Basic Cleaning
            t = token.strip().lstrip("-*").replace("®", "").replace("™", "").replace("*", "")
            
            #Handle [Active Ingredient] prefixes
            if ":" in t:
                parts = t.split(":", 1)
                lower_t = t.lower()
                t = parts[1].strip() if lower_t.startswith("active") or lower_t.startswith("ingredient") else parts[0].strip()
            
            #Remove content inside brackets/parentheses and percentages
            t = re.sub(r'[\(\[].*?[\)\]]', '', t)
            t = re.sub(r'\d+(?:\.\d+)?\s*%', '', t)
            t = t.strip()
            
            #Final Validation check
            if len(t) >= 2 and re.search(r'[a-zA-Z]', t):
                lower_t = t.lower()
                
                if lower_t not in self.IGNORE_LIST and not re.match(r'^ci\s?\d{5}$', lower_t):
                    cleaned_tokens.append(lower_t)
        
        # Deduplicate while preserving order
        return ", ".join(list(dict.fromkeys(cleaned_tokens)))

    def triggerRetraining(self):
        try:
            print("\n" + "="*40)
            print("[RETRAIN START] System Syncing...")
            
            db_url = f"{self.directus_url}/items/product?fields=id,name,ingredients.ingredient_id.name,suitable_skin_type&limit=-1"
            db_res = requests.get(db_url, headers=self.headers)
            db_data = db_res.json().get('data', [])
            print(f"Step 1/5: Downloaded {len(db_data)} products from DB.")

            os.makedirs("data", exist_ok=True)
            csv_path = "data/cosmetics_cleaned_final.csv"
            if self.csv_id:
                asset_res = requests.get(f"{self.directus_url}/assets/{self.csv_id}", headers=self.headers)
                df = pd.read_csv(StringIO(asset_res.text)) if asset_res.status_code == 200 else pd.DataFrame()
            else:
                df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

            print("Step 2/5: Processing and cleaning data...")
            for item in db_data:
                d_id = item.get('id')
                name = item.get('name', '').strip()
                ing_raw = ", ".join([i['ingredient_id']['name'] for i in item.get('ingredients', []) if i.get('ingredient_id')])
                
                clean_ing = self._cleanIngredients(ing_raw)
                skin_types = item.get('suitable_skin_type', [])
                
                mask = df['directus_id'] == d_id if 'directus_id' in df.columns else pd.Series([False]*len(df))
                if mask.any():
                    df.loc[mask, ['product_name', 'ingredients', 'clean_ingredients', 'skin_type']] = [name, ing_raw, clean_ing, str(skin_types)]
                else:
                    new_row = {"directus_id": d_id, "product_name": name, "ingredients": ing_raw, "clean_ingredients": clean_ing, "skin_type": str(skin_types)}
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            df.to_csv(csv_path, index=False)

            print("Step 3/5: Running SkinTypeClusteringService...")
            df_labeled, cluster_p = self.cluster_service.findClusterForSkinType(df)
            
            print("Step 4/5: Running IngredientAssociationService...")
            assoc_r = self.assoc_service.findRelatedIngredient(df_labeled)

            print("Step 5/5: Retraining completed successfully.")
            return df_labeled, cluster_p, assoc_r
        except Exception as e:
            print(f"❌ [RETRAIN ERROR] Pipeline failed: {e}")
            raise e