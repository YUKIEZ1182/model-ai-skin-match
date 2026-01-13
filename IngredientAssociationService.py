import pandas as pd
import os
import pickle
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

class IngredientAssociationService:
    def __init__(self):
        self.model_directory = "model"
        self.support_threshold = float(os.getenv("AI_SUPPORT", "0.02"))
        self.confidence_threshold = float(os.getenv("AI_CONFIDENCE", "0.5"))
        self.lift_threshold = float(os.getenv("AI_LIFT", "1.0"))
        self.maximum_length = int(os.getenv("AI_MAX_LEN", "3"))
        self.stop_ingredients_list = ['water', 'glycerin', 'phenoxyethanol', 'butylene glycol', 'aqua', 'alcohol']
        os.makedirs(self.model_directory, exist_ok=True)

    def findRelatedIngredient(self, dataframe):
        print(f"[ASSOCIATION] Mining started (Support: {self.support_threshold}, Confidence: {self.confidence_threshold})")
        
        ingredient_transactions = []
        for ingredients_string in dataframe["clean_ingredients"].dropna():
            items = [item.strip().lower() for item in ingredients_string.split(",") if item.strip()]
            filtered_items = [item for item in items if item not in self.stop_ingredients_list]
            if filtered_items: 
                ingredient_transactions.append(filtered_items)
        
        if not ingredient_transactions: 
            print("[ASSOCIATION] Warning: No transactions found to process.")
            return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift']), self.support_threshold, self.confidence_threshold, self.lift_threshold

        transaction_encoder = TransactionEncoder()
        encoded_array = transaction_encoder.fit(ingredient_transactions).transform(ingredient_transactions)
        encoded_dataframe = pd.DataFrame(encoded_array, columns=transaction_encoder.columns_)
        
        frequent_itemsets = apriori(
            encoded_dataframe, 
            min_support=self.support_threshold, 
            use_colnames=True, 
            max_len=self.maximum_length
        )
        
        if frequent_itemsets.empty:
            print("[ASSOCIATION] Warning: No frequent itemsets identified.")
            return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift']), self.support_threshold, self.confidence_threshold, self.lift_threshold

        generated_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=self.lift_threshold)
        
        if generated_rules.empty:
            print("[ASSOCIATION] Warning: No association rules generated.")
            return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift']), self.support_threshold, self.confidence_threshold, self.lift_threshold

        filtered_rules = generated_rules[generated_rules['confidence'] >= self.confidence_threshold].copy()

        if filtered_rules.empty:
            print("[ASSOCIATION] Warning: No rules remaining after confidence filtering.")
            return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift']), self.support_threshold, self.confidence_threshold, self.lift_threshold

        filtered_rules['antecedents'] = filtered_rules['antecedents'].apply(list)
        filtered_rules['consequents'] = filtered_rules['consequents'].apply(list)
        
        final_rules = filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(
            by=["lift", "confidence"], ascending=False
        )

        model_storage_path = os.path.join(self.model_directory, "association_model.pkl")
        with open(model_storage_path, "wb") as model_file:
            pickle.dump(final_rules, model_file)
            
        print(f"[ASSOCIATION] Completed. Total rules: {len(final_rules)}")
        return final_rules, self.support_threshold, self.confidence_threshold, self.lift_threshold

    def generate_plot(self, rules_dataframe):
        if rules_dataframe is None or rules_dataframe.empty: 
            print("[ASSOCIATION] Warning: No rules available for visualization.")
            return None
        
        try:
            top_performing_rules = rules_dataframe.nlargest(10, 'lift').copy()

            top_performing_rules['rule_label'] = (
                top_performing_rules['antecedents'].apply(lambda x: ", ".join(list(x))) + 
                " -> " + 
                top_performing_rules['consequents'].apply(lambda x: ", ".join(list(x)))
            )
            
            plt.figure(figsize=(12, 6))
            sns.barplot(data=top_performing_rules, x='lift', y='rule_label', palette='magma')
            plt.title('Top 10 Strongest Ingredient Associations (by Lift)', fontsize=15)
            plt.xlabel('Lift (Correlation Strength)')
            plt.ylabel('Ingredient Association Rules')
            
            plot_buffer = BytesIO()
            plt.savefig(plot_buffer, format='png', bbox_inches='tight')
            plt.close()
            plot_buffer.seek(0)
            return plot_buffer
        except Exception as error:
            print(f"[ASSOCIATION] Visualization Error: {error}")
            return None