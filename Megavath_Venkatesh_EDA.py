import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

def preprocess_data():
    # Load the datasets
    customers_df = pd.read_csv('Customers.csv')
    products_df = pd.read_csv('Products.csv')
    transactions_df = pd.read_csv('Transactions.csv')
    
    # Convert date columns to datetime
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
    
    # Merge datasets for analysis
    merged_df = transactions_df.merge(customers_df, on='CustomerID')
    merged_df = merged_df.merge(products_df, on='ProductID')
    
    return merged_df

def perform_eda(merged_df):
    # 1. Basic statistics
    print("\nBasic Statistics:")
    print(merged_df.describe())
    
    # 2. Sales by Region
    region_sales = merged_df.groupby('Region')['TotalValue'].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    region_sales.plot(kind='bar')
    plt.title('Total Sales by Region')
    plt.tight_layout()
    plt.savefig('sales_by_region.png')
    plt.close()
    
    # 3. Top selling products
    top_products = merged_df.groupby('ProductName')['Quantity'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    top_products.plot(kind='bar')
    plt.title('Top 10 Selling Products')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('top_products.png')
    plt.close()
    
    # 4. Customer purchase frequency
    customer_frequency = merged_df.groupby('CustomerID').size().describe()
    print("\nCustomer Purchase Frequency:")
    print(customer_frequency)
    
    # 5. Sales trends over time
    monthly_sales = merged_df.groupby(merged_df['TransactionDate'].dt.to_period('M'))['TotalValue'].sum()
    plt.figure(figsize=(12, 6))
    monthly_sales.plot(kind='line')
    plt.title('Monthly Sales Trend')
    plt.tight_layout()
    plt.savefig('sales_trend.png')
    plt.close()
    
    return {
        'region_sales': region_sales,
        'top_products': top_products,
        'customer_frequency': customer_frequency,
        'monthly_sales': monthly_sales
    }

def generate_business_insights(eda_results):
    # Get the top region and its sales
    top_region = eda_results['region_sales'].index[0]
    top_region_sales = eda_results['region_sales'].values[0]
    
    # Get the top product and its quantity
    top_product = eda_results['top_products'].index[0]
    top_product_quantity = eda_results['top_products'].values[0]
    
    insights = [
        f"Regional Performance: The {top_region} region leads in sales with ${top_region_sales:,.2f}, indicating strong market presence.",
        f"Product Success: The top-selling product is {top_product} with {top_product_quantity:,.0f} units sold.",
        f"Customer Behavior: Average purchase frequency per customer is {eda_results['customer_frequency']['mean']:.2f} transactions.",
        f"Sales Trend: Monthly sales show {'positive' if eda_results['monthly_sales'].diff().mean() > 0 else 'negative'} growth trend.",
        "Category Distribution: Analysis of product categories reveals opportunities for inventory optimization and targeted marketing campaigns."
    ]
    
    return insights

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    merged_df = preprocess_data()
    
    print("Performing exploratory data analysis...")
    eda_results = perform_eda(merged_df)
    
    print("Generating business insights...")
  

# Generate business insights
print("Generating business insights...")
insights = generate_business_insights(eda_results)

# Create a PDF document
print("Saving insights to PDF...")

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Set title
pdf.set_font("Arial", size=16, style='B')
pdf.cell(200, 10, txt="Business Insights Report", ln=True, align='C')
pdf.ln(10)  # Line break

# Set font for content
pdf.set_font("Arial", size=12)

# Write insights to PDF
for i, insight in enumerate(insights, 1):
    pdf.multi_cell(0, 10, f"{i}. {insight}\n")
    pdf.ln(5)  # Add a little spacing between insights

# Save the PDF
pdf.output("Megavath_Venkatesh_EDA.pdf")

print("Insights have been saved to 'Megavath_Venkatesh_EDA.pdf'.")
