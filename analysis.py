import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Load the dataset
file_path = 'crypto_all_62coins_daily.csv'
cols = ['timeOpen', 'timeClose', 'timeHigh', 'timeLow', 'name', 'open', 'high', 'low', 'close', 'volume', 'marketCap', 'circulatingSupply', 'timestamp']

# Read the data with semicolon delimiter
df = pd.read_csv(file_path, sep=';', names=cols, header=0)

# Convert numeric fields
numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'marketCap', 'circulatingSupply']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Create a date column from timeOpen
df['date'] = pd.to_datetime(df['timeOpen']).dt.date

# Sort by date
df = df.sort_values('date').reset_index(drop=True)

# Compute daily returns

df['return'] = df['close'].pct_change()

# Outlier detection for returns
returns_z = zscore(df['return'].dropna())
returns_z_full = pd.Series(index=df.index, dtype=float)
returns_z_full.iloc[1:] = returns_z
df['return_z'] = returns_z_full

# Outlier detection for volume using log transformation
log_volume = np.log1p(df['volume'])
vol_z = zscore(log_volume.dropna())
z_vol_full = pd.Series(index=df.index, dtype=float)
z_vol_full.loc[log_volume.notnull()] = vol_z
df['volume_z'] = z_vol_full

# Weighted PCA using standardized features
features = ['open', 'high', 'low', 'close', 'volume', 'marketCap']
weights = df['volume'].fillna(0).values + 1e-8
W_sum = weights.sum()

X = df[features].values
mean_w = np.sum(X * weights.reshape(-1, 1), axis=0) / W_sum
variance_w = np.sum(weights.reshape(-1, 1) * (X - mean_w) ** 2, axis=0) / W_sum
std_w = np.sqrt(variance_w)
X_scaled = (X - mean_w) / std_w

# Compute weighted covariance matrix
cov_w = np.dot((X_scaled * weights.reshape(-1, 1)).T, X_scaled) / W_sum
vals, vecs = np.linalg.eig(cov_w)
order = np.argsort(vals)[::-1]
vals = vals[order]
vecs = vecs[:, order]

explained_variance_ratio = vals / vals.sum()
PC1_scores = np.dot(X_scaled, vecs[:, 0])

df['PC1'] = PC1_scores

# Weighted regression of returns on PC1
returns = df['return'].fillna(0).values
mean_return = np.sum(weights * returns) / W_sum
mean_PC1 = np.sum(weights * PC1_scores) / W_sum
cov_xy = np.sum(weights * (PC1_scores - mean_PC1) * (returns - mean_return)) / W_sum
var_x = np.sum(weights * (PC1_scores - mean_PC1) ** 2) / W_sum
beta = cov_xy / var_x
intercept = mean_return - beta * mean_PC1
pred = intercept + beta * PC1_scores
ss_total = np.sum(weights * (returns - mean_return) ** 2)
ss_res = np.sum(weights * (returns - pred) ** 2)
R_squared = 1 - ss_res / ss_total

# Prepare output directory for plots
os.makedirs('plots', exist_ok=True)

# Plot distribution of returns
plt.figure(figsize=(8, 4))
plt.hist(df['return'].dropna(), bins=100, color='skyblue', edgecolor='black')
plt.title('Distribution of daily returns')
plt.xlabel('Daily return')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('plots/returns_distribution.png')

# Plot volume Z-scores
plt.figure(figsize=(10, 4))
plt.plot(df['date'], df['volume_z'], label='Log-volume z-score', color='green')
plt.axhline(3, color='red', linestyle='--', label='Outlier threshold')
plt.axhline(-3, color='red', linestyle='--')
plt.title('Volume z-scores over time')
plt.xlabel('Date')
plt.ylabel('Z-score')
plt.legend()
plt.tight_layout()
plt.savefig('plots/volume_zscore.png')

# Plot PC1 scores over time
plt.figure(figsize=(10, 4))
plt.plot(df['date'], df['PC1'], label='PC1 score')
plt.title('First principal component scores (weighted)')
plt.xlabel('Date')
plt.ylabel('PC1 score')
plt.tight_layout()
plt.savefig('plots/PC1_scores.png')

# Print summary statistics
print('Number of duplicate dates:', df['date'].duplicated().sum())
print('Number of return outliers:', (df['return_z'].abs() > 3).sum())
print('Number of volume outliers:', (df['volume_z'].abs() > 3).sum())
print('Explained variance ratio:', explained_variance_ratio)
print('PC1 loadings:', {features[i]: vecs[:, 0][i] for i in range(len(features))})
print('Correlation return-PC1:', cov_xy / np.sqrt(np.sum(weights * (returns - mean_return) ** 2) / W_sum * (var_x)))
print('Regression beta:', beta)
print('Regression intercept:', intercept)
print('Weighted R-squared:', R_squared)