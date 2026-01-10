import pandas as pd
import os
import pickle
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

class IngredientAssociationService:
    def __init__(self):
        self.ingredient = [] 
        self.MODEL_DIR = "model"
        self.MIN_SUPPORT = float(os.getenv("AI_MIN_SUPPORT", "0.02"))
        self.MAX_LEN = int(os.getenv("AI_MAX_LEN", "3"))
        self.MIN_LIFT = float(os.getenv("AI_MIN_LIFT", "1.0"))
        self.MIN_CONFIDENCE = float(os.getenv("AI_MIN_CONFIDENCE", "0.5"))
        
        os.makedirs(self.MODEL_DIR, exist_ok=True)

    def findRelatedIngredient(self, df):
        print(f"Starting Association Rule Mining (Support: {self.MIN_SUPPORT}, Max Len: {self.MAX_LEN})")
        
        stop_ingredients = ['water', 'glycerin', 'phenoxyethanol', 'butylene glycol', 'aqua', 'alcohol']
        
        transactions = []
        for x in df["clean_ingredients"].dropna():
            items = [item.strip().lower() for item in x.split(",") if item.strip()]
            filtered = [i for i in items if i not in stop_ingredients]
            if filtered: 
                transactions.append(filtered)
        
        self.ingredient = transactions 
        empty_df = pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
        
        if not transactions: 
            return empty_df

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        print(f"  > Mining frequent itemsets...", end=" ", flush=True)
        frequent_itemsets = apriori(df_encoded, min_support=self.MIN_SUPPORT, use_colnames=True, max_len=self.MAX_LEN)
        
        if frequent_itemsets.empty:
            return empty_df

        print(f"  > Generating rules...", end=" ", flush=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=self.MIN_LIFT)
        
        if rules.empty:
            return empty_df

        rules = rules[rules['confidence'] >= self.MIN_CONFIDENCE]

        if rules.empty:
            return empty_df

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