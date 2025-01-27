import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def create_customer_features(customers_df, transactions_df, products_df):
    # Convert date columns to datetime
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    
    # Calculate customer metrics
    customer_metrics = transactions_df.groupby('CustomerID').agg({
        'TotalValue': ['sum', 'mean', 'count'],
        'Quantity': ['sum', 'mean']
    }).reset_index()
    
    # Flatten column names
    customer_metrics.columns = ['CustomerID', 'total_spend', 'avg_transaction_value', 
                              'transaction_count', 'total_quantity', 'avg_quantity']
    
    # Calculate days since signup
    reference_date = pd.Timestamp('2025-01-27')
    customers_df['days_since_signup'] = (reference_date - customers_df['SignupDate']).dt.days
    
    # Create category preferences
    category_preferences = transactions_df.merge(products_df, on='ProductID')\
        .groupby(['CustomerID', 'Category'])['TotalValue'].sum()\
        .unstack(fill_value=0)
    
    # Convert region to numeric using one-hot encoding
    region_dummies = pd.get_dummies(customers_df['Region'], prefix='region')
    
    # Combine all features
    customer_features = customers_df[['CustomerID']].copy()
    customer_features['days_since_signup'] = customers_df['days_since_signup'].fillna(0)
    
    # Merge metrics and fill NaN values with 0
    customer_features = customer_features.merge(customer_metrics, on='CustomerID', how='left')
    numerical_cols = ['total_spend', 'avg_transaction_value', 'transaction_count', 
                     'total_quantity', 'avg_quantity']
    customer_features[numerical_cols] = customer_features[numerical_cols].fillna(0)
    
    # Add category preferences
    for col in category_preferences.columns:
        customer_features[f'category_{col}'] = customer_features['CustomerID'].map(
            category_preferences[col].to_dict()
        ).fillna(0)
    
    # Add region dummies
    for col in region_dummies.columns:
        customer_features[col] = customer_features['CustomerID'].map(
            dict(zip(customers_df['CustomerID'], region_dummies[col]))
        ).fillna(0)
    
    return customer_features

def find_lookalikes(customer_features, target_customer_id, n_recommendations=3):
    # Get numerical features (excluding CustomerID)
    numerical_features = customer_features.select_dtypes(include=[np.number]).columns.tolist()
    
    # Scale numerical features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_features[numerical_features])
    
    # Calculate similarity scores
    similarity_matrix = cosine_similarity(scaled_features)
    
    # Get index of target customer
    target_idx = customer_features[customer_features['CustomerID'] == target_customer_id].index[0]
    
    # Get similarity scores for target customer
    similarity_scores = similarity_matrix[target_idx]
    
    # Get indices of top similar customers (excluding self)
    similar_indices = np.argsort(similarity_scores)[::-1][1:n_recommendations+1]
    
    # Get customer IDs and similarity scores
    recommendations = [
        {
            'customer_id': customer_features.iloc[idx]['CustomerID'],
            'similarity_score': float(similarity_scores[idx])
        }
        for idx in similar_indices
    ]
    
    return recommendations

def save_recommendations_csv(recommendations_dict, output_file):
    rows = []
    for cust_id, recs in recommendations_dict.items():
        row = {
            'CustomerID': cust_id,
            'Lookalike1': recs[0]['customer_id'],
            'Score1': f"{recs[0]['similarity_score']:.4f}",
            'Lookalike2': recs[1]['customer_id'],
            'Score2': f"{recs[1]['similarity_score']:.4f}",
            'Lookalike3': recs[2]['customer_id'],
            'Score3': f"{recs[2]['similarity_score']:.4f}"
        }
        rows.append(row)
    
    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    print("Loading data...")
    customers_df = pd.read_csv('Customers.csv')
    products_df = pd.read_csv('Products.csv')
    transactions_df = pd.read_csv('Transactions.csv')
    
    print("Creating customer features...")
    customer_features = create_customer_features(customers_df, transactions_df, products_df)
    
    print("Generating lookalikes for first 20 customers...")
    lookalike_results = {}
    first_20_customers = customers_df['CustomerID'].iloc[:20]
    
    for cust_id in first_20_customers:
        print(f"Processing customer {cust_id}...")
        recommendations = find_lookalikes(customer_features, cust_id)
        lookalike_results[cust_id] = recommendations
    
    print("Saving results...")
    save_recommendations_csv(lookalike_results, 'Megavath_Venkatesh_Lookalike.csv')
    print("Done!")