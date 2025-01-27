import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def prepare_clustering_features(customers_df, transactions_df, products_df):
    """
    Prepare features for customer clustering using RFM analysis and transaction metrics.
    """
    # Ensure date columns are datetime
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
    
    # Calculate customer metrics
    customer_metrics = transactions_df.groupby('CustomerID').agg({
        'TotalValue': ['sum', 'mean', 'count'],
        'Quantity': ['sum', 'mean']
    }).reset_index()
    
    # Flatten column names
    customer_metrics.columns = ['CustomerID', 'total_spend', 'avg_transaction_value', 
                              'transaction_count', 'total_quantity', 'avg_quantity']
    
    # Calculate RFM metrics
    latest_date = transactions_df['TransactionDate'].max()
    rfm = transactions_df.groupby('CustomerID').agg({
        'TransactionDate': lambda x: (latest_date - pd.to_datetime(x.max())).days,  # Recency
        'TransactionID': 'count',  # Frequency
        'TotalValue': 'sum'  # Monetary
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'recency', 'frequency', 'monetary']
    
    # Combine features
    clustering_features = customer_metrics.merge(rfm, on='CustomerID')
    
    # Add category preferences
    category_preferences = transactions_df.merge(products_df, on='ProductID')\
        .groupby(['CustomerID', 'Category'])['TotalValue'].sum()\
        .unstack(fill_value=0)
    
    # Add category columns to features
    for category in category_preferences.columns:
        clustering_features[f'category_{category}'] = clustering_features['CustomerID'].map(
            category_preferences[category].to_dict()
        ).fillna(0)
    
    return clustering_features

def perform_clustering(features_df, min_clusters=2, max_clusters=10):
    """
    Perform K-means clustering and find optimal number of clusters using DB Index.
    """
    # Scale features
    scaler = StandardScaler()
    features_for_scaling = features_df.drop('CustomerID', axis=1)
    scaled_features = scaler.fit_transform(features_for_scaling)
    
    # Find optimal number of clusters using DB Index
    db_scores = []
    silhouette_scores = []
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        db_scores.append(davies_bouldin_score(scaled_features, clusters))
        silhouette_scores.append(silhouette_score(scaled_features, clusters))
        
        print(f"Clusters: {n_clusters}, DB Index: {db_scores[-1]:.4f}, Silhouette: {silhouette_scores[-1]:.4f}")
    
    # Select optimal number of clusters (minimum DB Index)
    optimal_clusters = db_scores.index(min(db_scores)) + min_clusters
    
    # Perform final clustering
    final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    clusters = final_kmeans.fit_predict(scaled_features)
    
    return {
        'clusters': clusters,
        'db_scores': db_scores,
        'silhouette_scores': silhouette_scores,
        'optimal_clusters': optimal_clusters,
        'scaled_features': scaled_features,
        'cluster_centers': final_kmeans.cluster_centers_
    }

def visualize_clusters(features_df, clustering_results):
    """
    Create visualizations for clustering results.
    """
    # Perform PCA for visualization
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(clustering_results['scaled_features'])
    
    # Create cluster visualization
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], 
                         c=clustering_results['clusters'], cmap='viridis')
    plt.title(f'Customer Segments (n_clusters={clustering_results["optimal_clusters"]})')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig('cluster_visualization.png')
    plt.close()
    
    # Plot DB Index scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, len(clustering_results['db_scores']) + 2), 
             clustering_results['db_scores'], marker='o')
    plt.title('Davies-Bouldin Index by Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Bouldin Index')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('db_index_scores.png')
    plt.close()

def generate_clustering_report(clustering_results, features_df):
    """
    Generate a detailed report of clustering results.
    """
    # Basic clustering metrics
    report_lines = [
        "Customer Segmentation Results",
        "============================\n",
        f"Number of Clusters: {clustering_results['optimal_clusters']}",
        f"Davies-Bouldin Index: {clustering_results['db_scores'][clustering_results['optimal_clusters']-2]:.4f}",
        f"Silhouette Score: {clustering_results['silhouette_scores'][clustering_results['optimal_clusters']-2]:.4f}\n",
        "Cluster Sizes:",
    ]
    
    # Add cluster sizes
    cluster_sizes = pd.Series(clustering_results['clusters']).value_counts().sort_index()
    for cluster_id, size in cluster_sizes.items():
        report_lines.append(f"Cluster {cluster_id}: {size} customers ({size/len(features_df)*100:.1f}%)")
    
    # Calculate cluster characteristics
    features_with_clusters = features_df.copy()
    features_with_clusters['Cluster'] = clustering_results['clusters']
    
    report_lines.append("\nCluster Characteristics:")
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    
    for cluster_id in range(clustering_results['optimal_clusters']):
        cluster_stats = features_with_clusters[features_with_clusters['Cluster'] == cluster_id][numeric_cols].mean()
        report_lines.append(f"\nCluster {cluster_id} Profile:")
        for col in ['total_spend', 'avg_transaction_value', 'transaction_count', 'recency']:
            if col in cluster_stats:
                report_lines.append(f"- {col}: {cluster_stats[col]:.2f}")
    
    # Save report
    with open('Megavath_Venkatesh_Clustering.txt', 'w') as f:
        f.write('\n'.join(report_lines))

def main():
    """
    Main function to run the clustering analysis.
    """
    print("Loading data...")
    customers_df = pd.read_csv('Customers.csv')
    products_df = pd.read_csv('Products.csv')
    transactions_df = pd.read_csv('Transactions.csv')
    
    print("Preparing features for clustering...")
    clustering_features = prepare_clustering_features(customers_df, transactions_df, products_df)
    
    print("Performing clustering analysis...")
    clustering_results = perform_clustering(clustering_features)
    
    print("Creating visualizations...")
    visualize_clusters(clustering_features, clustering_results)
    
    print("Generating clustering report...")
    generate_clustering_report(clustering_results, clustering_features)
    
    print("Analysis complete! Check the following files:")
    print("- cluster_visualization.png")
    print("- db_index_scores.png")
    print("- Megavath_Venkatesh_Clustering.txt")

if __name__ == "__main__":
    main()