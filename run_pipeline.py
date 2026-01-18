# AUDIT LOG
# 1. What this file is supposed to do:
# 2. What I understand:
# 3. What I don't understand:
# 4. One thing that can be improved:

import oracledb
import os
import array
from dotenv import load_dotenv
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.messages import HumanMessage

# 1. CONFIGURATION
# ----------------
load_dotenv()
my_key = os.getenv("COHERE_API_KEY")

os.environ["COHERE_API_KEY"] = my_key
DB_USER = "sys"
DB_PWD = "mypassword"
# NEW (DevOps Friendly)
# If 'DB_DSN' env var exists (Docker), use it. If not, default to localhost (Laptop).
DB_DSN = os.getenv("DB_DSN", "localhost:1521/FREEPDB1")

def get_diet_plan(user_region_input, ml_diagnosis_dict):
    """
    Args:
        user_region_input (str): User's location (e.g., "Mumbai")
        ml_diagnosis_dict (dict): The output from the ML model
    """
    
    # --- PHASE 1: RETRIEVAL (The "RAG" Part) ---
    print(f"🔍 Searching database for: '{user_region_input}'...")
    
    conn = oracledb.connect(user=DB_USER, password=DB_PWD, dsn=DB_DSN, mode=oracledb.SYSDBA)
    cursor = conn.cursor()
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    
    # Embed the user's query
    query_vec = array.array("d", embeddings.embed_query(user_region_input))
    
    # SQL: Find the closest diet plan using Vector Distance
    # We fetch the top 1 result because your regions are distinct
    cursor.execute("""
        SELECT content 
        FROM diet_vectors 
        ORDER BY VECTOR_DISTANCE(v_embedding, :qv, COSINE) ASC 
        FETCH FIRST 1 ROWS ONLY
    """, [query_vec])
    
    row = cursor.fetchone()
    retrieved_context = row[0] if row else "General healthy eating advice."
    
    cursor.close()
    conn.close()

    # --- PHASE 2: GENERATION (The LLM Part) ---
    print("🤖 Generating personalized advice...")
    chat = ChatCohere(model="command-r-08-2024", temperature=0.25)
    
    # This prompt connects the ML Diagnosis with the Retrieved Diet
    system_prompt = f"""
    You are a friendly, expert Nutritionist AI.
    
    --- USER HEALTH PROFILE ---
    Diagnosis: {ml_diagnosis_dict['diagnosis']}
    Risk Score: {ml_diagnosis_dict['risk_score']:.2%}
    Location: {user_region_input}
    
    --- KNOWLEDGE BASE (Regional Diet) ---
    {retrieved_context}
    
    --- YOUR INSTRUCTIONS ---
    The user has asked for a diet plan. Based on their diagnosis and region:
    1. Acknowledge their region enthusiastically.
    2. Propose a specific "Breakfast" "Lunch" and "Dinner" from the Knowledge Base options.
    3. WARN them about 1 specific high-sugar food from the 'Avoid' list.
    4. Keep the tone {ml_diagnosis_dict['tone']}.
    5. Keep the response concise (under 150 words).
    """
    
    response = chat.invoke([HumanMessage(content=system_prompt)])
    return response.content

# ==========================================
# SIMULATION (Testing the Pipeline)
# ==========================================

# Scenario 1: The user we processed earlier (Likely Healthy)
# ml_result = {
#     'risk_score': 0.2655, 
#     'diagnosis': 'Likely Healthy', 
#     'tone': 'Cheery, encouraging, and positive'
# }

# # Change this to test different regions! 
# # Try: "I live in Delhi" or "I am from Chennai" or "Thane"
# user_input = "I live in Thane, Maharashtra" 

# print("\n" + "="*50)
# print(f"🩺 INPUT: {ml_result['diagnosis']} | 📍 REGION: {user_input}")
# print("="*50)

# final_advice = get_diet_plan(user_input, ml_result)

# print("\n🤖 FINAL AI RESPONSE:\n")
# print(final_advice)
# print("\n" + "="*50)