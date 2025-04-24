import pandas as pd
from google.colab import files
uploaded = files.upload()
import pandas as pd
df = pd.read_csv("bank-full.csv")
print(df.info())
print(df.head())
print(df.describe(include='all'))

import pandas as pd

df = pd.read_csv('/content/bank-full.csv', sep=';')

# Binary conversion
binary_cols = ['default', 'housing', 'loan', 'y']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# One-hot encoding for nominal categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

# Feature scaling for numeric features
from sklearn.preprocessing import StandardScaler

numerical_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

df.head()

#FEATURE SELECTION
#CORRELATION WITH TARGET

import seaborn as sns
import matplotlib.pyplot as plt

# Correlation heatmap
corr = df_encoded.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr[['y']].sort_values(by='y', ascending=False), annot=True)

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2

# Use MinMaxScaler to make all values non-negative
scaler = MinMaxScaler()
X_minmax = scaler.fit_transform(X)

# Chi2 Feature Selection
selector = SelectKBest(score_func=chi2, k=10)
fit = selector.fit(X_minmax, y)

# Create DataFrame with scores
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Score': fit.scores_
}).sort_values(by='Chi2 Score', ascending=False)

# Show Top 10
print("Top 10 Features by Chi2:")
display(feature_scores.head(10))

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)

importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title("Top 10 Important Features (Random Forest)")
plt.show()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm')
plt.title("PCA Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

    
