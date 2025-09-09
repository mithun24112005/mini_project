# Save this file as final_project_solution.py
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

# --- 2. Feature Engineering with PERFECT 32-bit Logic ---
def get_golden_result(row):
    a = row['a']
    b = row['b']
    opcode = row['opcode']
    
    # Use a 32-bit mask to simulate hardware behavior
    mask = 0xFFFFFFFF 

    if opcode == 0: # ADD
        result = (a + b) & mask
    elif opcode == 1: # SUB
        result = (a - b) & mask
    elif opcode == 2: # AND
        result = (a & b) & mask
    elif opcode == 3: # OR
        result = (a | b) & mask
    elif opcode == 4: # XOR
        result = (a ^ b) & mask
    else:
        result = 0

    # Convert the unsigned 32-bit result back to a signed integer
    if result > 0x7FFFFFFF:
        result -= 0x100000000
    return result

print("Performing feature engineering with correct 32-bit logic...")
df['golden_result'] = df.apply(get_golden_result, axis=1)
df['difference'] = (df['golden_result'] - df['faulty_result']).abs()

# --- 3. Data Preprocessing (No Scaling) ---
X = df[['a', 'b', 'opcode', 'difference']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data preprocessing complete.")

# --- 4. Model Training ---
print("Training the final Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Training complete.")

# --- 5. Final Validation with Expanded Custom Data ---
print("\n--- Running Final Validation ---")
test_data = {
    'a':             [12345678, 255, 42, 2000000000, 98765, 2147483647, -2147483648, 170, 9999, 55555, 43, 1000000000],
    'b':             [0, 65280, 42, 1000000000, 98765, -1, 1, 85, 9999, -1, 42, 1000000000],
    'opcode':        [2, 3, 1, 0, 4, 0, 1, 2, 3, 4, 1, 1],
    'faulty_result': [0, 65534, 0, -1294967295, 0, 2147483646, 2147483646, 0, 26383, -55556, 0, 0]
}
custom_df = pd.DataFrame(test_data)

# Apply the CORRECT feature engineering
custom_df['golden_result'] = custom_df.apply(get_golden_result, axis=1)
custom_df['difference'] = (custom_df['golden_result'] - custom_df['faulty_result']).abs()

# Prepare the custom data for the model
X_custom = custom_df[['a', 'b', 'opcode', 'difference']]

# Make final predictions
predictions = rf_model.predict(X_custom)

# Display final results
results_df = custom_df[['a', 'b', 'opcode', 'faulty_result']].copy()
results_df['Model Prediction'] = predictions
results_df['Expected Outcome'] = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]

print(results_df)