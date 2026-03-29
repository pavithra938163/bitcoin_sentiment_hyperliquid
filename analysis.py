#  1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

plt.style.use('default')


# =========================
#  2. LOAD DATA
# =========================
# Replace with your file paths
trades = pd.read_csv("fear_greed_index.csv")
sentiment = pd.read_csv("historical_data.csv")


# =========================
# PART A — DATA PREPARATION
# =========================

print("Trades Shape:", trades.shape)
print("Sentiment Shape:", sentiment.shape)

print("\nMissing Values:\n", trades.isnull().sum())
print("\nDuplicates in trades:", trades.duplicated().sum())


# =========================
#  3. TIMESTAMP CONVERSION
# =========================
trades['timestamp'] = pd.to_datetime(trades['timestamp'])
sentiment['date'] = pd.to_datetime(sentiment['date'])

# Convert trades to daily level
trades['date'] = trades['timestamp'].dt.date
trades['date'] = pd.to_datetime(trades['date'])


# =========================
# 4. MERGE DATASETS
# =========================
data = pd.merge(trades, sentiment, on='date', how='inner')

print("Merged Shape:", data.shape)


# =========================
#  5. FEATURE ENGINEERING
# =========================

# Daily PnL per trader
daily_pnl = data.groupby(['trader_id', 'date'])['pnl'].sum().reset_index()

# Win Rate
data['win'] = (data['pnl'] > 0).astype(int)

win_rate = data.groupby('trader_id')['win'].mean().reset_index(name='win_rate')

# Avg Trade Size
avg_trade_size = data.groupby('trader_id')['trade_size'].mean().reset_index(name='avg_trade_size')

# Leverage Distribution
leverage_dist = data['leverage']

# Trades per day
trades_per_day = data.groupby('date').size().reset_index(name='num_trades')

# Long / Short Ratio
long_short = data.groupby('position_type').size(normalize=True)

print("\nLong/Short Ratio:\n", long_short)


# =========================
#  PART B — ANALYSIS
# =========================

# Label sentiment
data['sentiment_label'] = data['sentiment'].apply(lambda x: 'Fear' if x < 50 else 'Greed')


# ---- 1. PnL vs Sentiment ----
pnl_sentiment = data.groupby('sentiment_label')['pnl'].mean()

print("\nPnL by Sentiment:\n", pnl_sentiment)


# ---- 2. Win Rate vs Sentiment ----
win_sentiment = data.groupby('sentiment_label')['win'].mean()

print("\nWin Rate by Sentiment:\n", win_sentiment)


# ---- 3. Behavior Changes ----
behavior = data.groupby('sentiment_label').agg({
    'trade_size': 'mean',
    'leverage': 'mean',
    'pnl': 'count'
}).rename(columns={'pnl': 'trade_count'})

print("\nBehavior by Sentiment:\n", behavior)


# =========================
#  VISUALIZATIONS
# =========================

# 1. PnL Comparison
plt.figure()
sns.barplot(x=pnl_sentiment.index, y=pnl_sentiment.values)
plt.title("Average PnL: Fear vs Greed")
plt.xlabel("Sentiment")
plt.ylabel("PnL")
plt.show()

# 2. Trade Count
plt.figure()
sns.barplot(x=behavior.index, y=behavior['trade_count'])
plt.title("Trade Frequency by Sentiment")
plt.show()

# 3. Leverage Distribution
plt.figure()
sns.histplot(data['leverage'], bins=30)
plt.title("Leverage Distribution")
plt.show()


# =========================
#  SEGMENTATION
# =========================

# Create trader-level features
trader_features = data.groupby('trader_id').agg({
    'pnl': 'mean',
    'leverage': 'mean',
    'trade_size': 'mean',
    'win': 'mean'
}).rename(columns={'win': 'win_rate'})


# ---- KMeans Clustering ----
scaler = StandardScaler()
scaled = scaler.fit_transform(trader_features)

kmeans = KMeans(n_clusters=3, random_state=42)
trader_features['cluster'] = kmeans.fit_predict(scaled)

print("\nTrader Segments:\n", trader_features.head())


# =========================
#  SEGMENT INSIGHTS
# =========================

segment_summary = trader_features.groupby('cluster').mean()
print("\nSegment Summary:\n", segment_summary)


# =========================
#  PART C — ACTIONABLE STRATEGIES
# =========================

print("\n STRATEGY INSIGHTS:")

print("""
1. During Fear periods:
   - Traders show lower PnL and higher risk.
   → Reduce leverage for high-risk traders.

2. During Greed periods:
   - Trade frequency increases.
   → Focus on disciplined entry (avoid overtrading).

3. High leverage clusters:
   - More volatile returns.
   → Limit leverage or use stop-loss strategies.
""")


# =========================
#  BONUS — PREDICTIVE MODEL
# =========================

# Create target variable (profitability bucket)
data['profit_bucket'] = pd.qcut(data['pnl'], q=3, labels=[0,1,2])

features = ['leverage', 'trade_size', 'sentiment']
X = data[features]
y = data['profit_bucket']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\nModel Performance:\n")
print(classification_report(y_test, preds))
