import streamlit as st
import os
from dotenv import load_dotenv

# Import your custom modules
import extractor
import predictor
import run_pipeline

# 1. SETUP & CONFIGURATION
# ------------------------
st.set_page_config(page_title="Diabet-AI Assistant", page_icon="🩺", layout="wide")
load_dotenv()

if not os.getenv("COHERE_API_KEY"):
    st.error("❌ COHERE_API_KEY not found! Please check your .env file.")
    st.stop()

# Initialize Session State
if 'ml_result' not in st.session_state:
    st.session_state.ml_result = None
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
# New state to store the initial AI explanation
if 'initial_response' not in st.session_state:
    st.session_state.initial_response = None

# 2. UI HEADER
# ------------
st.title("🩺 AI Diabetes Assistant")
st.markdown("Analyze diabetes risk using **Machine Learning** and get personalized Indian diet plans via **Oracle Vector DB**.")
st.markdown("---")

# 3. INPUT SECTION (Tabs)
# -----------------------
tab1, tab2 = st.tabs(["📝 Manual Entry", "Vm Report Upload"])

# --- TAB 1: MANUAL ENTRY ---
with tab1:
    with st.form("manual_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            g = st.number_input("Glucose", 0, 500, 140)
            bp = st.number_input("Blood Pressure", 0, 200, 80)
        with col2:
            i = st.number_input("Insulin", 0, 900, 15)
            age = st.number_input("Age", 0, 120, 25)
        with col3:
            bmi = st.number_input("BMI", 0.0, 70.0, 28.0)
            preg = st.number_input("Pregnancies", 0, 20, 0)
        
        submit_manual = st.form_submit_button("Run Analysis")
        
        if submit_manual:
            input_list = [g, i, bp, age, bmi, preg]
            with st.spinner("Running ML Model..."):
                # A. Run Prediction
                result = predictor.predict_diabetes(input_list)
                st.session_state.ml_result = result
                st.session_state.extracted_data = input_list
                
                # B. Generate the Initial Explanation (USING YOUR FUNCTION)
                # This calls extractor.generate_initial_response
                ai_msg = extractor.generate_initial_response(result)
                
                # Extract text if it returns an object, or use as is
                st.session_state.initial_response = ai_msg.text if hasattr(ai_msg, 'text') else str(ai_msg)
                
                st.rerun()

# --- TAB 2: UPLOAD REPORT ---
with tab2:
    st.info("Upload a medical summary (PDF or Text).")
    uploaded_file = st.file_uploader("Upload Report", type=['txt', 'pdf'])
    
    if uploaded_file and st.button("Extract & Analyze"):
        with st.spinner("Extracting & Analyzing..."):
            pdf_obj = uploaded_file if uploaded_file.type == "application/pdf" else None
            text_obj = str(uploaded_file.read().decode("utf-8")) if uploaded_file.type == "text/plain" else None
            
            extracted_json = extractor.extracting_data(text_input=text_obj, pdf_file=pdf_obj)
            
            if extracted_json:
                data_list = [
                    extracted_json.get('Glucose', 0),
                    extracted_json.get('Insulin', 0),
                    extracted_json.get('BloodPressure', 0),
                    extracted_json.get('Age', 0),
                    extracted_json.get('BMI', 0),
                    extracted_json.get('Pregnancies', 0)
                ]
                
                # A. Prediction
                result = predictor.predict_diabetes(data_list)
                st.session_state.ml_result = result
                st.session_state.extracted_data = extracted_json
                
                # B. Initial Explanation (USING YOUR FUNCTION)
                ai_msg = extractor.generate_initial_response(result)
                st.session_state.initial_response = ai_msg.text if hasattr(ai_msg, 'text') else str(ai_msg)
                
                st.success("Complete!")
                st.rerun()
            else:
                st.error("Extraction failed.")

# 4. RESULTS DISPLAY (Corrected Flow)
# -----------------------------------
if st.session_state.ml_result:
    res = st.session_state.ml_result
    
    st.markdown("### 🔬 Analysis Results")
    
    # 1. Visual Metrics (The Raw Numbers)
    m1, m2 = st.columns([1, 3])
    with m1:
        st.metric("Risk Score", f"{res['risk_score']:.1%}")
        color = "green" if res['risk_score'] < 0.3 else "orange" if res['risk_score'] < 0.6 else "red"
        st.markdown(f":{color}[**{res['diagnosis']}**]")
        
    # 2. The AI Explanation (This comes from extractor.generate_initial_response)
    # It ends with: "Would you like suggestions... tell me which part of India..."
    with m2:
        st.info(f"**AI Assistant:**\n\n{st.session_state.initial_response}")

    # 3. The "Yes" Response (Diet Form)
    # This form now acts as the direct answer to the AI's question above.
    st.markdown("---")
    st.subheader("🥗 Regional Diet Plan")
    st.caption("Answer the assistant's question above to get your plan:")
    
    with st.form("diet_form"):
        region_input = st.text_input("Which part of India are you from?", placeholder="e.g. Thane, Maharashtra")
        get_diet_btn = st.form_submit_button("Get Diet Suggestions")
        
        if get_diet_btn and region_input:
            with st.spinner(f"Consulting Oracle DB for {region_input}..."):
                try:
                    # Call run_pipeline.py for the RAG part
                    diet_advice = run_pipeline.get_diet_plan(region_input, res)
                    st.success("Here is your personalized plan:")
                    st.markdown(f"""
                        <div style="background-color:#f0f2f6; color:#000000; padding:20px; border-radius:10px; border-left:5px solid #ff4b4b;">
                            {diet_advice}
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Pipeline Error: {e}")

# Reset Button
if st.session_state.ml_result:
    if st.button("Start Over"):
        st.session_state.ml_result = None
        st.session_state.extracted_data = None
        st.session_state.initial_response = None
        st.rerun()