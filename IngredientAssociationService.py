import pandas as pd
import os
import pickle
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

class IngredientAssociationService:
    def __init__(self):
        self.ingredient = [] 
        self.MODEL_DIR = "model"
        os.makedirs(self.MODEL_DIR, exist_ok=True)

    def findRelatedIngredient(self, df):
        print("Starting Association Rule Mining")
        
        stop_ingredients = ['water', 'glycerin', 'phenoxyethanol', 'butylene glycol', 'aqua', 'alcohol']
        
        transactions = []
        for x in df["clean_ingredients"].dropna():
            items = [item.strip().lower() for item in x.split(",") if item.strip()]
            filtered = [i for i in items if i not in stop_ingredients]
            if filtered: 
                transactions.append(filtered)
        
        self.ingredient = transactions 
        
        if not transactions: 
            return pd.DataFrame()

        print(f"  > Transactions prepared: {len(transactions)}")
        
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        print(f"  > Mining frequent itemsets (min_support=0.2)...", end=" ", flush=True)
        frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
        print(f"Found {len(frequent_itemsets)} itemsets.")

        print(f"  > Generating rules (lift threshold=1.2)...", end=" ", flush=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
        print(f"Generated {len(rules)} rules.")
        
        if rules.empty:
            return pd.DataFrame()

        rules['antecedents'] = rules['antecedents'].apply(list)
        rules['consequents'] = rules['consequents'].apply(list)
        rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(
            by=["lift", "confidence"], ascending=False
        )

        model_path = os.path.join(self.MODEL_DIR, "association_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(rules, f)
            
        print("Association Training Completed")
        return rules