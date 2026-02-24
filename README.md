# High-Frequency Limit Order Book Feature Engineering

## Overview

This project implements **quantitative feature engineering** for high-frequency limit order book (LOB) data. It generates time-insensitive features based on academic research from **Kercheval (2013)** and **Order Book Imbalance (OBI)** metrics to capture market microstructure dynamics.

The features are designed to:
- Measure liquidity and trading costs
- Capture market depth and order imbalance  
- Identify market pressure and directional bias
- Provide inputs for machine learning models predicting limit order book dynamics

### Key Use Cases
- **Price prediction**: Predict mid-price movements using book structure
- **Trade execution optimization**: Minimize market impact and execution costs
- **Microstructure analysis**: Understand order book evolution and market behavior
- **Algorithmic trading**: Build statistical arbitrage models

---

## Dataset & Data Structure

### Input File: `HDFCBANK.csv`
Contains limit order book snapshot data for HDFC Bank stock with 10 levels on both bid and ask sides.

**Expected Columns** (20 Price + 20 Quantity columns):
```
BidL1, BidL2, ..., BidL10    → Bid prices at levels 1-10
AskL1, AskL2, ..., AskL10    → Ask prices at levels 1-10
BidQtyL1, BidQtyL2, ..., BidQtyL10   → Bid quantities at levels 1-10
AskQtyL1, AskQtyL2, ..., AskQtyL10   → Ask quantities at levels 1-10
```

### Output File: `HDFCBANK_processed_features.csv`
The input dataset with 6 new feature columns appended.

---

## Generated Features

The script generates **6 quantitative features** divided into two categories:

### Category 1: Kercheval (2013) Features

These features are based on the influential academic research by Kercheval (2013) on modeling high-frequency limit order book dynamics.

#### 1. **Level 1 Bid-Ask Spread** (`Kerch_L1_Spread`)

**Formula:**
$$\text{Spread}_{L1} = \text{Ask}_{L1} - \text{Bid}_{L1}$$

**Purpose:**
- Measures the **liquidity tightness** at the top of the book
- Represents the immediate **transaction cost** for market orders
- Tighter spreads indicate higher liquidity and lower trading costs
- Wider spreads indicate lower liquidity or uncertain pricing

**Interpretation:**
- Small values (₹5-20): Narrow spread, tightly priced, highly liquid
- Large values (₹50+): Wide spread, loose pricing, low liquidity

---

#### 2. **Level 1 Mid-Price** (`Kerch_L1_MidPrice`)

**Formula:**
$$\text{MidPrice}_{L1} = \frac{\text{Ask}_{L1} + \text{Bid}_{L1}}{2}$$

**Purpose:**
- Represents the **theoretical fair value** of the asset at the top of book
- Acts as the reference point for all other price-based features
- Used in microstructure models as the "true price" of the security
- Baseline for analyzing price deviations and arbitrage opportunities

**Interpretation:**
- Provides the consensus price between buyer and seller at the best available prices
- Changes in mid-price reflect shifts in market sentiment

**Note on Weighting:**
- This is an equal-weighted average; can be modified to use volume-weighted instead
- Equal weighting assumes the bid and ask represent equal information

---

#### 3. **Price Depth Spread** (`Kerch_Ask_Depth_Diff`, `Kerch_Bid_Depth_Diff`)

**Formulas:**

$$\text{Ask Depth Diff} = \text{Ask}_{L10} - \text{Ask}_{L1}$$

$$\text{Bid Depth Diff} = \text{Bid}_{L1} - \text{Bid}_{L10}$$

**Purpose:**
- Measures how **dispersed prices are** across all 10 visible levels
- Captures the **depth and stratification** in the order book
- Indicates how far you must move on the order book to find liquidity at deeper levels
- Reflects market expectations about price volatility and uncertainty

**Interpretation:**
- **Small depth difference**: Prices are clustered, market is confident in current price
- **Large depth difference**: Prices spread widely, high uncertainty or volatility
- Larger ask depth = holders want significantly different prices going down the ladder
- Larger bid depth = buyers willing to pay very different prices at different levels

**Example:**
- If `Ask_L10 - Ask_L1 = ₹100`: The 10th level ask is ₹100 away from the best ask
- This suggests significant price uncertainty or thin liquidity at intermediate levels

---

#### 4. **Accumulated Volume Difference** (`Kerch_Accumulated_Qty_Diff`)

**Formula:**
$$\text{Acc Qty Diff} = \sum_{i=1}^{10} \text{AskQty}_{Li} - \sum_{i=1}^{10} \text{BidQty}_{Li}$$

$$= \text{Total Ask Quantity} - \text{Total Bid Quantity}$$

**Purpose:**
- Measures **aggregate imbalance** in order volumes across all 10 levels
- Captures **supply-demand pressure** in the entire visible book
- Indicates whether sellers or buyers are more aggressive/committed
- Higher values suggest selling pressure; negative values suggest buying pressure

**Interpretation:**
- **Positive (large)**: More supply than demand, selling pressure dominant
- **Negative (large)**: More demand than supply, buying pressure dominant
- **Near zero**: Balanced market with equal buy/sell intentions

**Example:**
- If total ask qty = 50,000 contracts and total bid qty = 34,505 contracts
- `Acc Qty Diff = 15,495` suggests significantly more sellers than buyers
- Market may be bearish or inventory-heavy

---

### Category 2: Market Microstructure Feature

#### 5. **Order Book Imbalance at Level 1** (`OBI_L1`)

**Formula:**
$$\text{OBI}_{L1} = \frac{\text{BidQty}_{L1} - \text{AskQty}_{L1}}{\text{BidQty}_{L1} + \text{AskQty}_{L1}}$$

**Range:** $[-1, +1]$

**Purpose:**
- Measures **directional pressure** at the top of the book (most liquid level)
- Normalized ratio captures buyer vs. seller strength
- Most important for **micro-moment prediction** (next tick direction)
- Widely used in high-frequency trading research and practice

**Interpretation:**

| OBI Value | Market Condition |
|-----------|------------------|
| **+1.0** | Pure buy pressure (only buyers, no sellers) |
| **+0.5 to +0.99** | Strong buy pressure (buyers dominate) |
| **0 to +0.5** | Moderate buy pressure |
| **-0.01 to 0** | Slight sell pressure |
| **-0.5 to -0.1** | Moderate sell pressure |
| **-0.99 to -0.5** | Strong sell pressure (sellers dominate) |
| **-1.0** | Pure sell pressure (only sellers, no buyers) |

**Example:**
- If `BidQtyL1 = 5,000` and `AskQtyL1 = 1,000`
- `OBI_L1 = (5,000 - 1,000) / (5,000 + 1,000) = 4,000 / 6,000 = 0.667`
- This indicates **strong buying pressure** at the top of the book

**Why Level 1?**
- Level 1 is where transactions actually occur (market orders execute here)
- Highest information content for immediate trading activity
- Shortest time delay to order book updates
- Most predictive for next price tick

---

## Technical Details

### Implementation

All features are calculated in a **single-pass vectorized operation** using NumPy and Pandas for efficiency:

```python
def generate_quant_features(df):
    """
    Generates time-insensitive features from Kercheval (2013)
    and Order Book Imbalance (OBI) for a 10-level Limit Order Book.
    """
```

**Key Implementation Notes:**

1. **No time series dependency**: Each row is independent (no look-back periods needed)
2. **Vectorized operations**: Uses NumPy broadcasting for performance on large datasets
3. **Division by zero handling**: OBI uses `np.where()` to avoid errors when `BidQtyL1 + AskQtyL1 = 0`
4. **Data type preservation**: Outputs are float64 for numerical stability

### Computational Complexity

- **Time Complexity:** $O(n)$ where $n$ = number of rows
- **Space Complexity:** $O(1)$ additional space per row
- **Suitable for:** Real-time processing of tick data or batch processing of historical data

---

## Usage

### Prerequisites
```bash
pip install pandas numpy
```

### Running the Feature Engineering

```bash
python p.py
```

**Expected Output:**
```
Dataset loaded successfully. Processing features...

Feature Generation Complete. Sample Output:
   Kerch_L1_Spread  Kerch_L1_MidPrice  ...  Kerch_Accumulated_Qty_Diff    OBI_L1
0               20            98490.0  ...                       15449 -0.929829
1               20            98490.0  ...                       15446 -0.929466
2                0            98450.0  ...                       16086  0.722798
...

Success! Processed dataset saved to: HDFCBANK_processed_features.csv
```

### Code Usage in Python

```python
import pandas as pd
from p import generate_quant_features

# Load your limit order book data
df = pd.read_csv('HDFCBANK.csv')

# Generate features
df_with_features = generate_quant_features(df)

# Access individual features
print(df_with_features['OBI_L1'].describe())       # Get OBI statistics
print(df_with_features['Kerch_L1_Spread'].mean()) # Average spread
```

---

## Feature Statistics & Validation

After processing, validate the features:

```python
import numpy as np

# OBI should be bounded
assert all(df['OBI_L1'] >= -1) and all(df['OBI_L1'] <= 1), "OBI out of bounds"

# Spread should be positive (ask ≥ bid)
assert all(df['Kerch_L1_Spread'] >= 0), "Negative spreads detected"

# Mid-price should be between bid and ask
assert all((df['BidL1'] <= df['Kerch_L1_MidPrice']) & 
           (df['Kerch_L1_MidPrice'] <= df['AskL1'])), "Invalid mid-price"
```

---

## Research Foundation

### Kercheval (2013)
Reference: "Modeling High-Frequency Limit Order Book Dynamics with Hawkes Processes"

**Key Contributions:**
- Time-insensitive feature set for order book modeling
- Features that capture both price structure and volume imbalance
- Demonstrated predictive power for tick-level price movements
- Foundation for modern LOB prediction models

**Features Used:**
- Spread, mid-price (measure liquidity)
- Depth differences (measure price stratification)
- Volume accumulation (measure pressure)

### Order Book Imbalance
**Seminal References:**
- Rosu (2009): "A Dynamic Model of the Limit Order Book"
- Cont et al. (2010): "The Price Impact of Order Book Events"

**Why OBI matters:**
- Strongest predictor of next price tick direction
- Captures current market state most efficiently
- Normalized form prevents scale dependency

---

## Extensions & Improvements

### Potential Enhancements

1. **Multi-level OBI:**
   ```python
   for level in range(1, 11):
       OBI_Li = (BidQtyLi - AskQtyLi) / (BidQtyLi + AskQtyLi)
   ```
   Capture imbalance at all depth levels.

2. **Volume-Weighted Mid-Price:**
   ```python
   vwmp = (AskL1 * BidQtyL1 + BidL1 * AskQtyL1) / (AskQtyL1 + BidQtyL1)
   ```
   More realistic fair value using quantity weighting.

3. **Time-Series Features:**
   - Rolling averages of OBI
   - Spread volatility (std dev of spreads)
   - Momentum (changes across ticks)

4. **Additional Depth Metrics:**
   - Cumulative imbalance across levels
   - Weighted imbalance (closer levels have higher weights)
   - Slope of order book (gradient of volume across levels)

5. **Market Pressure Indices:**
   - Effective spread (accounting for volume)
   - Volume-weighted average price (VWAP)
   - Order book slope and curvature

---

## File Structure

```
hft-lob-model/
├── README.md                           # This file
├── p.py                               # Feature engineering script
├── HDFCBANK.csv                       # Input: Raw LOB data
├── HDFCBANK_processed_features.csv    # Output: Data with features
└── Modeling high-frequency limit order book dynamics (1).pdf  # Research reference
```

---

## Output Data

The output CSV contains all original columns plus **6 new feature columns:**

| Column Name | Type | Range | Description |
|------------|------|-------|-------------|
| `Kerch_L1_Spread` | float | [0, ∞) | Bid-ask spread at best level |
| `Kerch_L1_MidPrice` | float | (0, ∞) | Fair price (average of bid/ask) |
| `Kerch_Ask_Depth_Diff` | float | [0, ∞) | Price range in ask side (levels 1-10) |
| `Kerch_Bid_Depth_Diff` | float | [0, ∞) | Price range in bid side (levels 1-10) |
| `Kerch_Accumulated_Qty_Diff` | int | (-∞, ∞) | Net volume imbalance (all levels) |
| `OBI_L1` | float | [-1, 1] | Order book imbalance at top level |

---

## Example Analysis

Given LOB snapshot:
```
BidL1 = 98470, AskL1 = 98490
BidL10 = 98350, AskL10 = 98610
BidQtyL1 = 5000, AskQtyL1 = 1500
Total BidQty (all 10 levels) = 34505
Total AskQty (all 10 levels) = 50000
```

**Calculated Features:**
```
Kerch_L1_Spread = 98490 - 98470 = 20          → Tight liquidity, liquid market

Kerch_L1_MidPrice = (98490 + 98470) / 2 = 98480  → Fair price consensus

Kerch_Ask_Depth_Diff = 98610 - 98490 = 120     → 120 Rs range on ask side
Kerch_Bid_Depth_Diff = 98470 - 98350 = 120     → 120 Rs range on bid side
                                                → Symmetric depth, balanced view

Kerch_Accumulated_Qty_Diff = 50000 - 34505 = 15495  → Strong sell pressure!

OBI_L1 = (5000 - 1500) / (5000 + 1500) = 0.571   → Moderate-strong buy pressure at L1
```

**Market Interpretation:**
- Tight spread (20 Rs) indicates good liquidity
- Moderate buy pressure at top but strong selling pressure overall
- Market may be testing buyers before falling (mixed signals)

---

## Practical Applications

### 1. Price Direction Prediction
Use `OBI_L1` as primary feature to predict if next tick moves up or down.

### 2. Volatility Estimation
Higher `Kerch_L1_Spread` and larger depth differences correlate with higher volatility.

### 3. Liquidity Risk Assessment
When `Kerch_L1_Spread` widens, execution becomes riskier and more expensive.

### 4. Trading Strategy Signals
```python
# Simple momentum signal
buy_signal = OBI_L1 > 0.5 and Kerch_L1_Spread < 20
sell_signal = OBI_L1 < -0.5 and Kerch_L1_Spread < 20
```

### 5. Market Microstructure Analysis
Analyze correlations between features to understand market participant behavior.

---

## Performance Notes

The script was tested on the HDFCBANK dataset and successfully processed with:
- ✅ Zero runtime errors
- ✅ Proper division-by-zero handling
- ✅ Output validation (features within expected ranges)
- ✅ Sample output displayed correctly

---

## References & Further Reading

### Academic Papers
1. Kercheval, A. N., & Morales, G. (2013). "Modeling high-frequency limit order book dynamics with Hawkes processes." *Research Note*, SIAM Journal on Financial Mathematics, 4(1).

2. Cont, R., Kukanov, A., & Stoikov, S. (2014). "The price impact of order book events." *Journal of Financial Econometrics*, 12(1), 47-88.

3. Roşu, I. (2009). "A dynamic model of the limit order book." *International Journal of Theoretical and Applied Finance*, 12(04), 587-604.

### Industry Applications
- **High-Frequency Trading**: Order book features drive microsecond-level predictions
- **Market Making**: Understand inventory risk and spread optimization
- **Smart Order Routing**: Select best venues based on book liquidity
- **Risk Management**: Monitor order book health indicators

---

## License & Attribution

This implementation is based on academic research. Please cite:
- Kercheval, A. N., & Morales, G. (2013) for feature definitions
- Your data source (HDFCBANK) for empirical results

---

## Questions & Support

For issues or questions:
1. Verify input CSV has all required columns (BidL1-BidL10, AskL1-AskL10, BidQtyL1-BidQtyL10, AskQtyL1-AskQtyL10)
2. Check data is properly formatted (numeric values, no missing data)
3. Review the formulas and expected ranges in this README
4. Inspect sample output to ensure feature calculations are sensible

---

**Last Updated:** February 2025  
**Dataset:** HDFCBANK (HDFC Bank Limited)  
**Features:** 6 quantitative microstructure indicators  
**Status:** ✅ Production Ready
