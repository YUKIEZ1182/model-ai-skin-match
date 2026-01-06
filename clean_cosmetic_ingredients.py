import pandas as pd
import re

# --- Helper Functions ---

IGNORE_LIST = {
    # 1. Water
    'water', 'aqua', 'eau', 'purified water', 'distilled water',
    
    # 2. Humectants & Solvents
    'glycerin', 'butylene glycol', 'propylene glycol', 'dipropylene glycol', 
    'propanediol', 'pentylene glycol', 'caprylyl glycol', 'hexylene glycol',
    '1,2-hexanediol', 'peg/ppg-17/6 copolymer',
    
    # 3. Alcohol & Fragrance
    'alcohol', 'alcohol denat', 'sd alcohol', 'ethanol',
    'fragrance', 'parfum', 'flavor', 'aroma', 
    'limonene', 'linalool', 'geraniol', 'citronellol', 'citral',
    
    # 4. Preservatives & pH Adjusters
    'phenoxyethanol', 'ethylhexylglycerin', 'sodium benzoate', 'potassium sorbate',
    'disodium edta', 'tetrasodium edta', 'trisodium edta',
    'citric acid', 'sodium hydroxide', 'potassium hydroxide', 'triethanolamine',
    'chlorphenesin', 'bht', 'tocopherol', 'tocopheryl acetate',
    
    # 5. Texture Enhancers & Silicones
    'carbomer', 'xanthan gum', 'acrylates/c10-30 alkyl acrylate crosspolymer',
    'dimethicone', 'cyclopentasiloxane', 'cyclohexasiloxane', 'dimethiconol',
    'stearic acid', 'palmitic acid', 'myristic acid', 'lauric acid',
    'cetyl alcohol', 'cetearyl alcohol', 'stearyl alcohol', 'behenyl alcohol',
    'glyceryl stearate', 'peg-100 stearate', 'polysorbate 20', 'polysorbate 60', 'polysorbate 80',
    'hydrogenated lecithin', 'polyacrylate crosspolymer-6'
}

def is_valid_ingredient(token):
    """
    Check if a token is a valid ingredient:
    - At least 2 characters.
    - Contains at least one alphabetical character.
    """
    token = token.strip()
    if len(token) < 2:
        return False
    if not re.search(r'[a-zA-Z]', token):
        return False
    return True

def clean_ingredient_token(token):
    """
    Clean an individual ingredient token.
    """
    token = token.strip()
    token = token.lstrip("-*")
    token = token.replace("®", "").replace("™", "").replace("*", "")
    
    if ":" in token:
        parts = token.split(":", 1)
        lower_token = token.lower()
        if lower_token.startswith("active ingredient") or lower_token.startswith("ingredient"):
            token = parts[1].strip()
        else:
            token = parts[0].strip()
    
    token = re.sub(r'\d+(?:\.\d+)?\s*%', '', token)
    return token.strip()

def clean_ingredients(ingredient_str):
    """
    Clean a full ingredient string and remove ignored ingredients.
    """
    if pd.isna(ingredient_str):
        return ""
    
    # Standardize delimiters
    ingredient_str = ingredient_str.replace(" and ", ",").replace(";", ",")
    tokens = ingredient_str.split(",")
    cleaned_tokens = []
    
    for token in tokens:
        cleaned = clean_ingredient_token(token)
        if is_valid_ingredient(cleaned):
            lower_cleaned = cleaned.lower()
            
            # ✅ เช็ค Ignore List ตรงนี้
            if lower_cleaned not in IGNORE_LIST:
                cleaned_tokens.append(lower_cleaned)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tokens = []
    for token in cleaned_tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)
    
    return ", ".join(unique_tokens)

# --- Main Processing ---

def main():
    # ตรวจสอบ path ไฟล์ให้ถูกต้องตามเครื่องของคุณ
    input_file = "data/cosmetic_base.csv" 
    output_file = "data/cosmetics_cleaned.csv"
    
    print(f"Reading from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        
        # เช็คว่ามีคอลัมน์ ingredients หรือไม่
        if "ingredients" not in df.columns:
            print(f"Error: Column 'ingredients' not found in {input_file}")
            return

        print("Cleaning ingredients...")
        df["clean_ingredients"] = df["ingredients"].apply(clean_ingredients)
        
        # ลบแถวที่ส่วนผสมหายไปหมดหลัง clean (เช่น สินค้าที่มีแต่น้ำกับน้ำหอม)
        df = df[df["clean_ingredients"] != ""]
        
        df.to_csv(output_file, index=False)
        print(f"✅ Cleaning complete. Saved to {output_file}")
        print(f"Removed common ingredients like Water, Glycerin, Phenoxyethanol, etc.")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found at {input_file}")

if __name__ == "__main__":
    main()