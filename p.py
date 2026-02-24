import pandas as pd
import numpy as np

def generate_quant_features(df):
    """
    Generates time-insensitive features from Kercheval (2013)
    and Order Book Imbalance (OBI) for a 10-level Limit Order Book.
    """
    # Create a copy to avoid SettingWithCopy warnings
    df = df.copy()

    # Define column lists for easy aggregation
    ask_qty_cols = [f'AskQtyL{i}' for i in range(1, 11)]
    bid_qty_cols = [f'BidQtyL{i}' for i in range(1, 11)]

    # ---------------------------------------------------------
    # FEATURES FROM KERCHEVAL (2013)
    # ---------------------------------------------------------

    # 1. Feature Set v2: Bid-Ask Spread at Level 1
    # Measures the immediate cost of trading and liquidity tightness.
    df['Kerch_L1_Spread'] = df['AskL1'] - df['BidL1']

    # 2. Feature Set v2: Mid-Price at Level 1
    # The theoretical fair value of the asset at the top of the book.
    df['Kerch_L1_MidPrice'] = (df['AskL1'] + df['BidL1']) / 2.0

    # 3. Feature Set v3: Price Differences (Depth Spread)
    # Measures how spread out the prices are across the visible 10 levels.
    df['Kerch_Ask_Depth_Diff'] = df['AskL10'] - df['AskL1']
    df['Kerch_Bid_Depth_Diff'] = df['BidL1'] - df['BidL10']

    # 4. Feature Set v5: Accumulated Volume Differences
    # Tracks the accumulated volume pressure between the ask and bid sides.
    total_ask_qty = df[ask_qty_cols].sum(axis=1)
    total_bid_qty = df[bid_qty_cols].sum(axis=1)
    df['Kerch_Accumulated_Qty_Diff'] = total_ask_qty - total_bid_qty

    # ---------------------------------------------------------
    # EXTERNAL FEATURE: Order Book Imbalance (OBI)
    # ---------------------------------------------------------
    
    # 5. Order Book Imbalance at Level 1 (Top of Book)
    # Formula: (BidQty - AskQty) / (BidQty + AskQty)
    # Ranges from -1 (heavy sell pressure) to +1 (heavy buy pressure).
    imbalance_num = df['BidQtyL1'] - df['AskQtyL1']
    imbalance_den = df['BidQtyL1'] + df['AskQtyL1']
    
    # Use np.where to handle division by zero safely
    df['OBI_L1'] = np.where(
        imbalance_den == 0, 
        0, 
        imbalance_num / imbalance_den
    )

    return df

if __name__ == "__main__":
    # Specify your input and output file paths
    input_file_path = 'HDFCBANK.csv'
    output_file_path = 'HDFCBANK_processed_features.csv'

    try:
        # 1. Load the Limit Order Book dataset
        raw_lob_df = pd.read_csv(input_file_path)
        print("Dataset loaded successfully. Processing features...")

        # 2. Apply the feature engineering function
        processed_lob_df = generate_quant_features(raw_lob_df)

        # 3. Display the newly generated features to verify
        new_columns = [col for col in processed_lob_df.columns if 'Kerch' in col or 'OBI' in col]
        print("\nFeature Generation Complete. Sample Output:")
        print(processed_lob_df[new_columns].head())

        # 4. Save the dataframe to a new CSV file
        processed_lob_df.to_csv(output_file_path, index=False)
        print(f"\nSuccess! Processed dataset saved to: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: The file at {input_file_path} was not found. Please ensure it is uploaded.")