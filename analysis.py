import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy import stats
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("bank-full.csv", delimiter=";")

# 1. Descriptive Analysis
## a) Aggregation & Summarization
print(df['job'].value_counts())
print(df['marital'].value_counts())
print(df['education'].value_counts())
print(df.groupby('job')['balance'].mean())
print(df['y'].value_counts())

## b) Data Cleaning & Preprocessing
print(df.isnull().sum())  # Check for missing
df.drop_duplicates(inplace=True)
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])
  

# 2. Diagnostic Analysis
## a) Root Cause Analysis
print(df.groupby('job')['y'].mean())
print(df.groupby(pd.cut(df['age'], bins=[0,30,50,100]))['y'].mean())
print(df.groupby('previous')['y'].mean())

## b) Hypothesis Testing
older = df[df['age'] > 50]['y']
younger = df[df['age'] <= 50]['y']
t_stat, p_val = stats.ttest_ind(older, younger)
print("T-test result:", t_stat, p_val)


# 3. Predictive Analysis
## a) Decision Tree Model
X = df.drop(['y'], axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree = DecisionTreeClassifier(max_depth=4)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))

plt.figure(figsize=(20,10))
plot_tree(tree, feature_names=X.columns, class_names=['No','Yes'], filled=True)
plt.show()

## b) Regression Analysis
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_log = logreg.predict(X_test)
print("Logistic Regression Report:\n", classification_report(y_test, y_pred_log))

# 4. Prescriptive Analysis
## a) A/B Testing (simulated example)
group_a = df[df['duration'] <= 180]['y']
group_b = df[df['duration'] > 300]['y']
t_stat_ab, p_val_ab = stats.ttest_ind(group_a, group_b)
print("A/B Testing T-test:", t_stat_ab, p_val_ab)

## b) DSS Simulation (Top 10 probable customers)
df['prob'] = logreg.predict_proba(X)[:, 1]
top_customers = df.sort_values(by='prob', ascending=False).head(10)
print("Top customers to target:\n", top_customers[['age', 'job', 'balance', 'duration', 'prob']])
