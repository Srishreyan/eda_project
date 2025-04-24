import pandas as pd
from google.colab import files

# Upload the file manually
uploaded = files.upload()

# Get the filename dynamically
file_path = list(uploaded.keys())[0]

# Load dataset
df = pd.read_csv(file_path)

# Generate frequency tables for categorical columns
frequency_tables = {col: df[col].value_counts() for col in df.select_dtypes(include=['object'])}

# Display frequency tables
for col, freq_table in frequency_tables.items():
    print(f"Frequency Table for {col}:\n{freq_table}\n")

# Example: Cross-tabulation between 'job' and 'marital' columns (Change columns as needed)
pivot_table = pd.crosstab(df['job'], df['marital'])

# Display pivot table
print("Cross-tabulation (Pivot Table):\n", pivot_table)

import numpy as np
# Example: Using 'job' column to model a Markov Chain
states = df['job'].astype(str).tolist()  # Convert to string for categorical data

# Compute transition matrix
unique_states = list(set(states))
state_index = {state: i for i, state in enumerate(unique_states)}

transition_matrix = np.zeros((len(unique_states), len(unique_states)))

for i in range(len(states) - 1):
    current_state = state_index[states[i]]
    next_state = state_index[states[i + 1]]
    transition_matrix[current_state, next_state] += 1

# Normalize each row to get probabilities
transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

# Convert to DataFrame for better readability
markov_chain = pd.DataFrame(transition_matrix, index=unique_states, columns=unique_states)

print("Markov Chain Transition Matrix:")
print(markov_chain)

# Compute Variance, Standard Deviation, and IQR for numerical columns
dispersion_measures = {
    "Variance": df.var(numeric_only=True),
    "Standard Deviation": df.std(numeric_only=True),
    "IQR": df.quantile(0.75, numeric_only=True) - df.quantile(0.25, numeric_only=True)
}

# Convert to DataFrame for better display
dispersion_df = pd.DataFrame(dispersion_measures)
print("Measures of Dispersion:\n", dispersion_df)

# Compute Mean, Median, Mode for each numerical column
central_tendency = {
    "Mean": df.mean(numeric_only=True),
    "Median": df.median(numeric_only=True),
    "Mode": df.mode(numeric_only=True).iloc[0]  # First mode in case of multiple
}

# Compute Mean, Median, Mode for each numerical column
central_tendency = {
    "Mean": df.mean(numeric_only=True),
    "Median": df.median(numeric_only=True),
    "Mode": df.mode(numeric_only=True).iloc[0]  # First mode in case of multiple
}

# Convert to DataFrame for better display
central_tendency_df = pd.DataFrame(central_tendency)
print("Measures of Central Tendency:\n", central_tendency_df)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder # Import LabelEncoder here
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Encode categorical variables
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].apply(le.fit_transform)
# Define features and target
X = df.drop(columns=['y'])  # Assuming 'y' is the target variable
y = df['y']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
svm_model = SVC()
rf_model = RandomForestClassifier()
nn_model = MLPClassifier(max_iter=500)

svm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
nn_model.fit(X_train, y_train)

# Predict & evaluate
models = {'SVM': svm_model, 'Random Forest': rf_model, 'Neural Network': nn_model}
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")

import numpy as np
from scipy.optimize import linprog

# Linear Programming Example
c = [-1, -2]  # Coefficients for the objective function (maximize x + 2y)
A = [[2, 1], [1, 2]]  # Coefficients for inequalities
b = [20, 20]  # Constraints
x_bounds = (0, None)  # x >= 0
y_bounds = (0, None)  # y >= 0

result = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, y_bounds], method='highs')
print("Linear Programming Result:")
print(result)

# Integer Programming Example
from scipy.optimize import minimize

def objective(x):
    return -(x[0] + 2 * x[1])  # Maximization problem

constraints = ({'type': 'ineq', 'fun': lambda x: 20 - (2*x[0] + x[1])},
               {'type': 'ineq', 'fun': lambda x: 20 - (x[0] + 2*x[1])})

bounds = [(0, None), (0, None)]  # Non-negative constraints

result_int = minimize(objective, [0, 0], bounds=bounds, constraints=constraints, method='SLSQP')
print("Integer Programming Result:")
print(result_int)


# Import the statsmodels library
import statsmodels.api as sm

# Encode categorical variables
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].apply(le.fit_transform)

# Define independent and dependent variables
X = df.drop(columns=['y'])  # Assuming 'y' is the target variable
y = df['y']

# Add constant for intercept
X = sm.add_constant(X) # Now sm is defined and can be used

# Fit regression model
model = sm.OLS(y, X).fit()

# Display statistical regression results
print("Statistical Regression Analysis:")
print(model.summary())

# Selecting only numerical columns
numerical_df = df.select_dtypes(include=['number'])

# Compute Pearson correlation
pearson_corr = numerical_df.corr(method='pearson')
print("Pearson Correlation Coefficients:\n", pearson_corr, "\n")

# Compute Spearman correlation
spearman_corr = numerical_df.corr(method='spearman')
print("Spearman Correlation Coefficients:\n", spearman_corr)
