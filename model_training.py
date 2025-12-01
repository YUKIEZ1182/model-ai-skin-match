import pandas as pd
import ast
import os
import pickle
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Ensure output directories exist
os.makedirs("model", exist_ok=True)
os.makedirs("csv_report", exist_ok=True)

def parse_literal_list(s):
    """
    Convert a string representation of a list or frozenset to an actual list.
    If the string starts with "frozenset(", it is removed.
    """
    if pd.isnull(s):
        return []
    s = s.strip()
    # Remove prefix "frozenset(" if present, and the trailing ")"
    if s.startswith("frozenset("):
        s = s[len("frozenset("):-1]
    try:
        result = ast.literal_eval(s)
        # If result is a set, convert it to list and sort for consistency
        if isinstance(result, set):
            return sorted(list(result))
        elif isinstance(result, list):
            return result
        else:
            return [result]
    except Exception:
        # Fallback: assume comma-separated string
        return [item.strip() for item in s.split(",") if item.strip()]

def mine_association_rules(cleaned_csv="data/cosmetics_cleaned.csv", min_support=0.05, min_confidence=0.7):
    print("Loading cleaned data from", cleaned_csv)
    df = pd.read_csv(cleaned_csv)
    
    print("Converting 'clean_ingredients' column into transactions...")
    transactions = (
        df["clean_ingredients"]
        .dropna()
        .apply(lambda x: [item.strip() for item in x.split(",") if item.strip()])
        .tolist()
    )
    print(f"Total transactions: {len(transactions)}")
    
    print("Performing one-hot encoding on transactions...")
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_ingredients = pd.DataFrame(te_ary, columns=te.columns_)
    
    print("Mining frequent itemsets with min_support =", min_support)
    frequent_itemsets = apriori(df_ingredients, min_support=min_support, use_colnames=True)
    print("Frequent itemsets found:", len(frequent_itemsets))
    
    print("Generating association rules with min_confidence =", min_confidence)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    print("Association rules generated:", len(rules))
    
    # Convert frozensets in antecedents and consequents to lists
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
    
    # Select only the desired columns and sort the DataFrame
    return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(
        by=["lift", "confidence", "support"], ascending=False
    )

def update_model(new_data, cleaned_csv="data/cosmetics_cleaned.csv", min_support=0.05, min_confidence=0.7):
    """
    Append new data to the existing cleaned data, re-mine association rules,
    and save the updated model and report.
    """
    if os.path.exists(cleaned_csv):
        print("Loading existing cleaned data...")
        old_data = pd.read_csv(cleaned_csv)
        combined_data = pd.concat([old_data, new_data], ignore_index=True)
    else:
        print("No existing data found. Creating new cleaned data.")
        combined_data = new_data

    combined_data.drop_duplicates(inplace=True)
    print("Saving combined cleaned data to", cleaned_csv)
    combined_data.to_csv(cleaned_csv, index=False)
    
    print("Mining association rules on updated data...")
    rules = mine_association_rules(cleaned_csv=cleaned_csv, min_support=min_support, min_confidence=min_confidence)
    
    # Save rules CSV into csv_report folder
    csv_path = os.path.join("csv_report", "association_rules.csv")
    print("Saving association rules to", csv_path)
    rules.to_csv(csv_path, index=False)
    
    # Save the model (rules DataFrame) into the model folder
    model_path = os.path.join("model", "association_model.pkl")
    print("Pickling the association rules model to", model_path)
    with open(model_path, "wb") as f:
        pickle.dump(rules, f)
    
    return combined_data, rules

def load_rules():
    """
    Load association rules from the CSV report and parse the antecedents and consequents.
    """
    csv_path = os.path.join("csv_report", "association_rules.csv")
    df = pd.read_csv(csv_path)
    df['antecedents'] = df['antecedents'].apply(parse_literal_list)
    df['consequents'] = df['consequents'].apply(parse_literal_list)
    return df.sort_values(by=["lift", "confidence", "support"], ascending=False)

def train_from_scratch(new_data, cleaned_csv="data/cosmetics_cleaned.csv", min_support=0.05, min_confidence=0.7):
    """
    Train a new model from scratch by overwriting existing cleaned data.
    """
    print("Training model from scratch...")
    new_data.drop_duplicates(inplace=True)
    print("Saving new cleaned data to", cleaned_csv)
    new_data.to_csv(cleaned_csv, index=False)
    
    print("Mining association rules from scratch...")
    rules = mine_association_rules(cleaned_csv=cleaned_csv, min_support=min_support, min_confidence=min_confidence)
    
    csv_path = os.path.join("csv_report", "association_rules.csv")
    print("Saving new association rules to", csv_path)
    rules.to_csv(csv_path, index=False)
    
    model_path = os.path.join("model", "association_model.pkl")
    print("Pickling new model to", model_path)
    with open(model_path, "wb") as f:
        pickle.dump(rules, f)
    
    return new_data, rules

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python model_training.py <path_to_new_data_csv>")
        sys.exit(1)
    
    new_data_file = sys.argv[1]
    try:
        print("Reading new data from", new_data_file)
        new_data = pd.read_csv(new_data_file)
        combined_data, rules = train_from_scratch(new_data)
        print(f"Trained new model from scratch with {len(combined_data)} transactions and {len(rules)} rules.")
    except Exception as e:
        print(f"Error during training: {e}")
