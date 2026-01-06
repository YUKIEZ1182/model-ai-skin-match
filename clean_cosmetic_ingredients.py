import pandas as pd
import re

# --- Helper Functions ---

IGNORE_LIST = {'water', 'aqua', 'eau', 'purified water', 'distilled water'}

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
            
            # ✅ เพิ่ม Logic เช็ค Ignore List ตรงนี้
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
    df = pd.read_csv("data/cosmetic_base.csv") # หรือไฟล์ที่คุณต้องการ test
    df["clean_ingredients"] = df["ingredients"].apply(clean_ingredients)
    df.to_csv("data/cosmetics_cleaned.csv", index=False)
    print("Cleaning complete. Water and others have been removed.")

if __name__ == "__main__":
    main()