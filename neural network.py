import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# STEP 1: Assume you have 'new_data' ready (features of future clients)
# For example, predicting 5 new clients
new_data = {
    'age': [30, 45, 35, 50, 40],
    'job': ['blue-collar', 'admin.', 'technician', 'retired', 'services'],
    'marital': ['married', 'single', 'married', 'married', 'single'],
    'education': ['secondary', 'tertiary', 'primary', 'secondary', 'tertiary'],
    'default': ['no', 'no', 'yes', 'no', 'no'],
    'housing': ['yes', 'yes', 'no', 'yes', 'no'],
    'loan': ['no', 'yes', 'yes', 'no', 'yes'],
    'contact': ['cellular', 'cellular', 'telephone', 'cellular', 'cellular'],
    'month': ['may', 'jul', 'oct', 'may', 'nov'],
    'day_of_week': ['mon', 'wed', 'fri', 'mon', 'thu'],
    'duration': [150, 200, 100, 180, 120],
    'campaign': [1, 2, 1, 2, 1],
    'pdays': [999, 999, 999, 999, 999],
    'previous': [0, 1, 0, 0, 2],
    'poutcome': ['nonexistent', 'failure', 'nonexistent', 'nonexistent', 'success']
}

# Convert new_data to DataFrame
new_df = pd.DataFrame(new_data)

# STEP 2: Preprocessing (encoding & scaling same as training)
new_df_encoded = pd.get_dummies(new_df)  # One-hot encode categorical
new_df_encoded = new_df_encoded.reindex(columns=X_encoded.columns, fill_value=0)  # Match training data columns
new_data_scaled = scaler.transform(new_df_encoded)  # Scale the new data

# STEP 3: Predict probabilities of subscription (1 or 0)
pred_probs = model.predict(new_data_scaled).ravel()

# STEP 4: Plot predicted probabilities
plt.figure(figsize=(10, 6))
plt.bar(range(len(pred_probs)), pred_probs, color='skyblue')
plt.title('Predicted Subscription Probabilities for Future Clients')
plt.xlabel('Client Index')
plt.ylabel('Predicted Probability of Subscription')
plt.xticks(range(len(pred_probs)), labels=[f'Client {i+1}' for i in range(len(pred_probs))])
plt.grid(True)
plt.show()
