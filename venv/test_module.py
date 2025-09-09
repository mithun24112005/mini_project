import pickle
import pandas as pd

# 1. Load your trained model
with open('D:\mini_project\venv\alu_fault_detector.pkl', "rb") as f:
    model = pickle.load(f)

# 2. Example test cases (like rows from your CSV dataset)
# Columns: a, b, opcode, golden_result, faulty_result
test_data = pd.DataFrame([
    # Case 1: No fault (OK)
    {"a": 10, "b": 20, "opcode": 0, "golden_result": 30, "faulty_result": 30},  # ADD 10+20=30

    # Case 2: Bit-flip fault
    {"a": 15, "b": 5, "opcode": 1, "golden_result": 10, "faulty_result": 8},   # SUB 15-5=10 → faulty_result wrong

    # Case 3: Stuck-at-0 fault
    {"a": 7, "b": 3, "opcode": 2, "golden_result": 3, "faulty_result": 0},     # AND 7&3=3 → but output stuck at 0

    # Case 4: Stuck-at-1 fault
    {"a": 6, "b": 2, "opcode": 3, "golden_result": 6, "faulty_result": 7},     # OR 6|2=6 → but output stuck at 1-bit

    # Case 5: Random XOR test
    {"a": 12, "b": 5, "opcode": 4, "golden_result": 9, "faulty_result": 11},   # XOR 12^5=9 → faulty_result wrong
])

# 3. Predict
predictions = model.predict(test_data)

# 4. Show results
for i, pred in enumerate(predictions):
    print(f"Test case {i+1}: Predicted label = {pred}")
