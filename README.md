# High-Frequency Limit Order Book Feature Engineering

## Overview

This project implements quantitative feature engineering for high-frequency limit order book (LOB) data. It generates time-insensitive features based on academic research from Kercheval & Zhang (2013) and Order Book Imbalance (OBI) metrics to capture market microstructure dynamics.

The features are designed to:
* Measure liquidity and trading costs
* Capture market depth and order imbalance  
* Identify market pressure and directional bias
* Provide inputs for machine learning models (like SVMs) predicting limit order book dynamics

### Key Use Cases
* **Price prediction**: Predict mid-price movements using book structure
* **Trade execution optimization**: Minimize market impact and execution costs
* **Microstructure analysis**: Understand order book evolution and market behavior
* **Algorithmic trading**: Build statistical arbitrage models

---

## Dataset & Data Structure

### Input File: `HDFCBANK.csv`
Contains limit order book snapshot data for HDFC Bank stock with 10 levels on both bid and ask sides.

**Expected Columns** (20 Price + 20 Quantity columns):
`BidL1`, `BidL2`, ..., `BidL10`    → Bid prices at levels 1-10
`AskL1`, `AskL2`, ..., `AskL10`    → Ask prices at levels 1-10
`BidQtyL1`, `BidQtyL2`, ..., `BidQtyL10`   → Bid quantities at levels 1-10
`AskQtyL1`, `AskQtyL2`, ..., `AskQtyL10`   → Ask quantities at levels 1-10

### Output File: `HDFCBANK_processed_features.csv`
The input dataset with 6 new feature columns appended.

---

## Generated Features

The script generates 6 quantitative features divided into two categories:

### Category 1: Kercheval & Zhang (2013) Features

These features are based on the influential academic research by Kercheval and Zhang (2013) on modeling high-frequency limit order book dynamics using Support Vector Machines.

#### 1. Level 1 Bid-Ask Spread (`Kerch_L1_Spread`)

**Formula:**
`Spread_L1 = Ask_L1 - Bid_L1`

**Purpose:**
* Measures the liquidity tightness at the top of the book.
* Represents the immediate transaction cost for market orders.
* Tighter spreads indicate higher liquidity and lower trading costs.
* Wider spreads indicate lower liquidity or uncertain pricing.

**Interpretation:**
* Small values (₹5-20): Narrow spread, tightly priced, highly liquid.
* Large values (₹50+): Wide spread, loose pricing, low liquidity.

---

#### 2. Level 1 Mid-Price (`Kerch_L1_MidPrice`)

**Formula:**
`MidPrice_L1 = (Ask_L1 + Bid_L1) / 2`

**Purpose:**
* Represents the theoretical fair value of the asset at the top of book.
* Acts as the reference point for all other price-based features.
* Used in microstructure models as the "true price" of the security.
* Baseline for analyzing price deviations and arbitrage opportunities.

**Interpretation:**
* Provides the consensus price between buyer and seller at the best available prices.
* Changes in mid-price reflect shifts in market sentiment.

**Note on Weighting:**
* This is an equal-weighted average; can be modified to use volume-weighted instead.
* Equal weighting assumes the bid and ask represent equal information.

---

#### 3. Price Depth Spread (`Kerch_Ask_Depth_Diff`, `Kerch_Bid_Depth_Diff`)

**Formulas:**
`Ask_Depth_Diff = Ask_L10 - Ask_L1`
`Bid_Depth_Diff = Bid_L1 - Bid_L10`

**Purpose:**
* Measures how dispersed prices are across all 10 visible levels.
* Captures the depth and stratification in the order book.
* Indicates how far you must move on the order book to find liquidity at deeper levels.
* Reflects market expectations about price volatility and uncertainty.

**Interpretation:**
* **Small depth difference**: Prices are clustered, market is confident in current price.
* **Large depth difference**: Prices spread widely, high uncertainty or volatility.
* Larger ask depth = holders want significantly different prices going down the ladder.
* Larger bid depth = buyers willing to pay very different prices at different levels.

**Example:**
* If `Ask_L10 - Ask_L1 = ₹100`: The 10th level ask is ₹100 away from the best ask.
* This suggests significant price uncertainty or thin liquidity at intermediate levels.

---

#### 4. Accumulated Volume Difference (`Kerch_Accumulated_Qty_Diff`)

**Formula:**
`Acc_Qty_Diff = Total_Ask_Quantity - Total_Bid_Quantity`

**Purpose:**
* Measures aggregate imbalance in order volumes across all 10 levels.
* Captures supply-demand pressure in the entire visible book.
* Indicates whether sellers or buyers are more aggressive/committed.
* Higher values suggest selling pressure; negative values suggest buying pressure.

**Interpretation:**
* **Positive (large)**: More supply than demand, selling pressure dominant.
* **Negative (large)**: More demand than supply, buying pressure dominant.
* **Near zero**: Balanced market with equal buy/sell intentions.

**Example:**
* If total ask qty = 50,000 contracts and total bid qty = 34,505 contracts.
* `Acc_Qty_Diff = 15,495` suggests significantly more sellers than buyers.
* Market may be bearish or inventory-heavy.

---

### Category 2: Market Microstructure Feature

#### 5. Order Book Imbalance at Level 1 (`OBI_L1`)

**Formula:**
`OBI_L1 = (BidQty_L1 - AskQty_L1) / (BidQty_L1 + AskQty_L1)`

**Range:** [-1, +1]

**Purpose:**
* Measures directional pressure at the top of the book (most liquid level).
* Normalized ratio captures buyer vs. seller strength.
* Most important for micro-moment prediction (next tick direction).
* Widely used in high-frequency trading research and practice.

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
* If `BidQty_L1 = 5,000` and `AskQty_L1 = 1,000`.
* `OBI_L1 = (5,000 - 1,000) / (5,000 + 1,000) = 4,000 / 6,000 = 0.667`.
* This indicates **strong buying pressure** at the top of the book.

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
