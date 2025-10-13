# Save this file as final_test.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- 1. Load Data ---
try:
    df = pd.read_csv('venv/dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'dataset.csv' not found. Make sure it's in the same folder.")
    exit()

# --- 2. Feature Engineering ---
def get_golden_result(row):
    a = row['a']
    b = row['b']
    opcode = row['opcode']
    if opcode == 0: return a + b
    if opcode == 1: return a - b
    if opcode == 2: return a & b
    if opcode == 3: return a | b
    if opcode == 4: return a ^ b
    return 0

print("Performing feature engineering...")
df['golden_result'] = df.apply(get_golden_result, axis=1)
df['difference'] = (df['golden_result'] - df['faulty_result']).abs()

# --- 3. Data Preprocessing (No Scaling) ---
X = df[['a', 'b', 'opcode', 'difference']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data preprocessing complete (without scaling).")

# --- 4. Model Training ---
print("Training the final Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) # Train on unscaled data
print("Training complete.")

# --- 5. Evaluate on Test Set ---
print("\n--- Model Performance on Standard Test Set ---")
y_pred_rf = rf_model.predict(X_test) # Predict on unscaled data
print(classification_report(y_test, y_pred_rf))

# --- 6. Final Validation with Custom Data ---
print("\n--- Running Final Validation with Custom Test Cases ---")
test_data = {
    'a': [100, 100, -1, 2000, 1024],
    'b': [50, 50, -1, 3000, 1024],
    'opcode': [0, 0, 4, 1, 2],
    'faulty_result': [150, 151, 0, -1001, 1024]
}
custom_df = pd.DataFrame(test_data)

# Apply the SAME feature engineering
custom_df['golden_result'] = custom_df.apply(get_golden_result, axis=1)
custom_df['difference'] = (custom_df['golden_result'] - custom_df['faulty_result']).abs()

# Prepare the custom data for the model (No scaling)
X_custom = custom_df[['a', 'b', 'opcode', 'difference']]

# Make final predictions
predictions = rf_model.predict(X_custom) # Predict on unscaled data

# Display final results
results_df = custom_df[['a', 'b', 'opcode', 'faulty_result']].copy()
results_df['Model Prediction'] = predictions
results_df['Expected Outcome'] = [0, 1, 0, 1, 0]

print(results_df)