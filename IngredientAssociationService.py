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
        # Load thresholds from environment variables
        self.support_threshold = float(os.getenv("AI_SUPPORT", "0.02"))
        self.confidence_threshold = float(os.getenv("AI_CONFIDENCE", "0.5"))
        self.lift_threshold = float(os.getenv("AI_LIFT", "1.0"))
        self.maximum_length = int(os.getenv("AI_MAX_LEN", "3"))
        
        # Ingredients to ignore to reduce noise in rules
        self.stop_ingredients_list = [
            'water', 'glycerin', 'phenoxyethanol', 'butylene glycol', 'aqua', 'alcohol',
            'sodium chloride', 'ethylhexylglycerin', 'disodium edta', 'sodium benzoate'
        ]
        os.makedirs(self.model_directory, exist_ok=True)

    def findRelatedIngredient(self, dataframe):
        print(f"[ASSOCIATION] Mining started. Support: {self.support_threshold}, Confidence: {self.confidence_threshold}")
        
        # 1. Prepare transactions from cleaned ingredients
        ingredient_transactions = []
        for ingredients_string in dataframe["clean_ingredients"].dropna():
            items = [item.strip().lower() for item in ingredients_string.split(",") if item.strip()]
            filtered_items = [item for item in items if item not in self.stop_ingredients_list]
            if filtered_items: 
                ingredient_transactions.append(filtered_items)
        
        if not ingredient_transactions: 
            print("[ASSOCIATION] Error: No valid transactions found to process.")
            return pd.DataFrame(), 0.0, 0.0, 0.0, None

        # 2. Encode transactions into binary matrix
        transaction_encoder = TransactionEncoder()
        encoded_array = transaction_encoder.fit(ingredient_transactions).transform(ingredient_transactions)
        encoded_dataframe = pd.DataFrame(encoded_array, columns=transaction_encoder.columns_)
        
        # 3. Apply Apriori algorithm
        frequent_itemsets = apriori(
            encoded_dataframe, 
            min_support=self.support_threshold, 
            use_colnames=True, 
            max_len=self.maximum_length
        )
        
        if frequent_itemsets.empty:
            print("[ASSOCIATION] Error: No frequent itemsets identified.")
            return pd.DataFrame(), 0.0, 0.0, 0.0, None

        # 4. Generate Association Rules
        generated_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=self.lift_threshold)
        
        if generated_rules.empty:
            print("[ASSOCIATION] Error: No rules generated with current lift threshold.")
            return pd.DataFrame(), 0.0, 0.0, 0.0, None

        # Filter by Confidence threshold
        filtered_rules = generated_rules[generated_rules['confidence'] >= self.confidence_threshold].copy()

        if filtered_rules.empty:
            print("[ASSOCIATION] Error: No rules remaining after confidence filtering.")
            return pd.DataFrame(), 0.0, 0.0, 0.0, None

        # 5. Calculate Match Strength for API visualization
        # Normalize Lift score (assuming 5.0 as 100% strength for UI purposes)
        filtered_rules['match_strength'] = filtered_rules['lift'].apply(
            lambda x: min(100, round((x / 5.0) * 100))
        )

        # Convert FrozenSets to Lists for JSON compatibility
        filtered_rules['antecedents'] = filtered_rules['antecedents'].apply(list)
        filtered_rules['consequents'] = filtered_rules['consequents'].apply(list)
        
        # Select and Sort final rules
        final_rules = filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'match_strength']].sort_values(
            by=["lift", "confidence"], ascending=False
        )

        # 6. Save final model to local storage
        model_storage_path = os.path.join(self.model_directory, "association_model.pkl")
        with open(model_storage_path, "wb") as model_file:
            pickle.dump(final_rules, model_file)

        # 7. Calculate average metrics for database logging
        avg_support = final_rules['support'].mean()
        avg_confidence = final_rules['confidence'].mean()
        avg_lift = final_rules['lift'].mean()
        total_rules = len(final_rules)
            
        print(f"[ASSOCIATION] Process completed. Rules found: {total_rules}, Avg Lift: {avg_lift:.4f}")
        
        # Generate visual plot
        plot_buffer = self.generate_plot(final_rules)
        
        # Return 5 values to match logging requirements
        return final_rules, avg_support, avg_confidence, avg_lift, plot_buffer

    def generate_plot(self, rules_dataframe):
        if rules_dataframe is None or rules_dataframe.empty: 
            return None
        
        try:
            # Select top 10 rules for visualization
            top_rules = rules_dataframe.nlargest(10, 'lift').copy()
            
            # Create readable labels (e.g., "Retinol + Niacinamide")
            top_rules['rule_name'] = (
                top_rules['antecedents'].apply(lambda x: x[0]) + 
                " + " + 
                top_rules['consequents'].apply(lambda x: x[0])
            )

            plt.figure(figsize=(10, 6))
            sns.barplot(data=top_rules, x='lift', y='rule_name', palette='viridis')
            
            plt.title('Top 10 Strongest Ingredient Associations', fontsize=14)
            plt.xlabel('Strength (Lift Score)', fontsize=12)
            plt.ylabel('Ingredient Pairs', fontsize=12)
            plt.tight_layout()
            
            plot_buffer = BytesIO()
            plt.savefig(plot_buffer, format='png', bbox_inches='tight')
            plt.close()
            plot_buffer.seek(0)
            return plot_buffer
        except Exception as error:
            print(f"[ASSOCIATION] Plot generation failed: {error}")
            return None