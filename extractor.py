import cohere
from PIL import Image
import PyPDF2
from dotenv import load_dotenv
import json
import os

load_dotenv()

#setting up Cohere

my_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(my_key)


#extracting data
def extracting_data(text_input=None, image_file=None, pdf_file=None):
    #extracts medical data from text image or pdf file

    #base prompt
    system_instruction = """
    You are a medical data extractor. Analyse the input text and extract these 6 parameters:
    - Glucose
    - Insulin
    - BloodPressure (BP)
    - Age
    - BMI
    - Pregnancies

    rules:
    1. Return ONLY a valid JSON object. Keys must exactly be: "Glucose","Insulin","BloodPressure","Age","BMI","Pregnancies".
    2. If a value is missing, use 0 as its value.
    """
    user_content = ""

    if text_input:
        user_content += f"\n\nPatient Description:\n{text_input}"

    if image_file:
        print("⚠️ Warning: Cohere API (Command-R) is text-only. It cannot read images directly. Skipping image.")

    if pdf_file:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pdf_text=""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()+"\n" or ""

            user_content += f"\n\nPDF Content:\n{pdf_text}"

        except Exception as e:
            print(f"Failed to read PDF file: {str(e)}")
        
    if not user_content:
        return None

    #calling cohere
    try:
        response = co.chat(
            model='command-r-08-2024',
            message = user_content,
            preamble=system_instruction,
            temperature=0.0
        )
        cleaned_text = response.text.replace("```json","").replace("```","").strip()
        data_dict = json.loads(cleaned_text)
        return data_dict

    except Exception as e:
        print(f"Error during data extraction: {str(e)}")
        return None


#initial response by LLM after ML prediction

def generate_initial_response(ml_result):
    # Extract variables
    score = ml_result['risk_score']
    diagnosis = ml_result['diagnosis']
    tone = ml_result['tone']
    rec = ml_result['recommendation']

    system_prompt = f"""
    You are a helpful AI health assistant.
    Current Context: The user has just finished a diabetes risk assessment.
    
    Data:
    - Risk Probability: {score:.2%}
    - Diagnosis: {diagnosis}
    - Recommendation: {rec}
    
    Instructions:
    1. Explain the results to the user using this tone: {tone}.
    2. Keep it concise (under 3 sentences).
    3. CRITICAL: You MUST end your response with exactly this question: 
       "Would you like personalized diet and lifestyle suggestions based on your region in India? If yes, please tell me which part of India you are from."
    """
    
    # Calling LLM here (Cohere) with this prompt
    response = co.chat(
            model='command-r-08-2024',
            message=system_prompt,
            temperature=0.0
        ) 
    
    return response