import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import io

# Set page config
st.set_page_config(
    page_title="ALU Fault Detector",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title(" Fault Detection System")
st.markdown("""
This app detects faults using a trained CNN model.
Upload a CSV file or enter values manually to test the model.
""")

# Load the trained model (make sure your model file is in the same directory)
@st.cache_resource
def load_trained_model():
    try:
        model = load_model('alu_fault_cnn_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_trained_model()

# Fault type mapping
FAULT_NAMES = {0: "✅ No Fault", 1: "⚠️ Bitflip", 2: "❌ Opcode Fault"}

# Helper functions
def calculate_correct_result(a, b, opcode):
    """Calculate correct ALU result based on opcode"""
    if opcode == 0: return (a + b) & 0xFFFFFFFF
    elif opcode == 1: return (a - b) & 0xFFFFFFFF
    elif opcode == 2: return a & b
    elif opcode == 3: return a | b
    elif opcode == 4: return a ^ b
    else: return 0

def error_to_binary(error_int):
    """Convert error integer to 32-bit binary sequence"""
    binary_str = format(error_int, '032b')
    return [int(bit) for bit in binary_str]

def predict_fault(a, b, opcode, faulty_result):
    """Predict fault type for given inputs"""
    correct = calculate_correct_result(a, b, opcode)
    error_bits = faulty_result ^ correct
    error_sequence = error_to_binary(error_bits)
    X_input = np.array(error_sequence).reshape(1, 32, 1)
    
    prediction = model.predict(X_input, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100
    
    return FAULT_NAMES[predicted_class], confidence, error_bits, correct

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose input method:", 
                           [" Manual Input", " Upload CSV"])

if app_mode == "📝 Manual Input":
    st.header("Manual Input Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        a = st.number_input("Operand A", value=100, min_value=0, max_value=2**32-1)
        b = st.number_input("Operand B", value=200, min_value=0, max_value=2**32-1)
    
    with col2:
        opcode = st.selectbox(
            "Operation",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: ["ADD", "SUB", "AND", "OR", "XOR"][x]
        )
        faulty_result = st.number_input("Faulty Result", value=300, min_value=0, max_value=2**32-1)
    
    if st.button("🔍 Detect Fault"):
        if model:
            fault_type, confidence, error_bits, correct_result = predict_fault(a, b, opcode, faulty_result)
            
            # Display results
            st.subheader("Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Expected Result", correct_result)
            
            with col2:
                st.metric("Faulty Result", faulty_result)
            
            with col3:
                st.metric("Detection", fault_type)
            
            # Confidence and details
            st.progress(int(confidence))
            st.write(f"Confidence: {confidence:.2f}%")
            
            # Error bits visualization
            st.subheader("Error Analysis")
            error_binary = format(error_bits, '032b')
            st.write(f"Error Bits: `{error_binary}`")
            st.write(f"Bits Flipped: {bin(error_bits).count('1')}")
            
            # Color-coded error bits
            error_display = ""
            for i, bit in enumerate(error_binary):
                if bit == '1':
                    error_display += f"<span style='color: red; font-weight: bold;'>{bit}</span>"
                else:
                    error_display += f"<span style='color: green;'>{bit}</span>"
                if (i + 1) % 8 == 0:
                    error_display += " "
            
            st.markdown(f"Visual: {error_display}", unsafe_allow_html=True)
            
        else:
            st.error("Model not loaded. Please check if 'alu_fault_cnn_model.h5' exists.")

else:  # CSV Upload mode
    st.header("Batch Testing with CSV Upload")
    
    st.markdown("""
    **Expected CSV format:**
    - Columns: `a`, `b`, `opcode`, `faulty_result`
    - Opcodes: 0=ADD, 1=SUB, 2=AND, 3=OR, 4=XOR
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.success(f" Successfully loaded {len(df)} records")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            if st.button("🚀 Process All Records"):
                if model:
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, row in df.iterrows():
                        a = row['a']
                        b = row['b']
                        opcode = row['opcode']
                        faulty = row['faulty_result']
                        
                        fault_type, confidence, error_bits, correct = predict_fault(a, b, opcode, faulty)
                        
                        results.append({
                            'A': a,
                            'B': b,
                            'Opcode': opcode,
                            'Expected_Result': correct,
                            'Faulty_Result': faulty,
                            'Predicted_Fault': fault_type,
                            'Confidence': f"{confidence:.2f}%",
                            'Bits_Flipped': bin(error_bits).count('1')
                        })
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(df))
                    
                    # Create results dataframe
                    results_df = pd.DataFrame(results)
                    
                    # Display results
                    st.subheader("Detection Results")
                    st.dataframe(results_df)
                    
                    # Summary statistics
                    st.subheader("Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    fault_counts = results_df['Predicted_Fault'].value_counts()
                    
                    with col1:
                        st.metric("Total Records", len(results_df))
                    
                    with col2:
                        no_fault_count = len([x for x in results_df['Predicted_Fault'] if "No Fault" in x])
                        st.metric("No Fault", no_fault_count)
                    
                    with col3:
                        fault_count = len(results_df) - no_fault_count
                        st.metric("Faults Detected", fault_count)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="fault_detection_results.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.error("Model not loaded. Please check if 'alu_fault_cnn_model.h5' exists.")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Footer
st.markdown("---")