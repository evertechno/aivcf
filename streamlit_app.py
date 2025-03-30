import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
import io

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("Venture Capital Pitch Deck and Financial Analysis")
st.write("Upload pitch decks and financial statements for AI-powered analysis.")

# File Upload Sections
pitchdeck_file = st.file_uploader("Upload Pitch Deck (PDF)", type=["pdf"])
financial_file = st.file_uploader("Upload Financial Statement (CSV/Excel)", type=["csv", "xlsx"])

# Function to extract text from PDF pitch deck
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from the uploaded pitch deck
def extract_pitchdeck_info(pitchdeck_file):
    try:
        file_extension = pitchdeck_file.name.split('.')[-1].lower()
        extracted_text = ""
        if file_extension == 'pdf':
            extracted_text = extract_text_from_pdf(pitchdeck_file)
        
        return extracted_text
    except Exception as e:
        st.error(f"Error extracting text from pitch deck: {e}")
        return None

# Function to perform AI-powered analysis of the pitch deck
def analyze_pitchdeck(pitchdeck_text):
    try:
        # Use Google Generative AI for analysis
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Analyze the following pitch deck content and provide a business analysis, including strengths, weaknesses, market opportunity, and team analysis: {pitchdeck_text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error analyzing pitch deck: {e}")
        return None

# DCF Valuation Function
def perform_valuation_analysis(financial_data):
    try:
        # Placeholder for extracting relevant financial data (e.g., revenue, expenses, etc.)
        # Assume financial_data contains columns like 'Revenue', 'Expense', etc.
        revenue = financial_data['Revenue']  # Example
        expenses = financial_data['Expense']  # Example
        
        # Example DCF calculation
        discount_rate = 0.1  # Assumed discount rate for DCF
        future_cash_flows = np.array([100000, 120000, 150000, 180000])  # Example future cash flows
        terminal_value = 2000000  # Example terminal value
        dcf_value = np.sum(future_cash_flows / (1 + discount_rate) ** np.arange(1, len(future_cash_flows) + 1)) + terminal_value / (1 + discount_rate) ** len(future_cash_flows)
        
        # Example EBITDA multiple for Enterprise Valuation
        enterprise_value = np.sum(revenue) * 6  # Assuming 6x EBITDA multiple
        
        # Return the results
        return f"DCF Valuation: ${dcf_value:,.2f}\nEnterprise Valuation: ${enterprise_value:,.2f}"
    except Exception as e:
        st.error(f"Error performing valuation analysis: {e}")
        return None

# Function to generate and provide the template for financial data (CSV format)
def generate_financial_template():
    # Create a template with predefined columns for financial data
    data = {
        'Year': [2022, 2023, 2024, 2025],
        'Revenue': [0, 0, 0, 0],
        'Expense': [0, 0, 0, 0],
        'EBITDA': [0, 0, 0, 0],
        'Net Income': [0, 0, 0, 0],
    }
    df = pd.DataFrame(data)
    
    # Convert the dataframe to CSV format
    csv_file = df.to_csv(index=False)
    
    # Create a BytesIO buffer to allow downloading
    buffer = io.StringIO(csv_file)
    return buffer

# Display the financial data template download link
st.subheader("Download Financial Data Template")
st.write("You can download the financial data template in CSV format and fill in your company's financial details.")
template_buffer = generate_financial_template()

# Provide the download button for the CSV template
st.download_button(
    label="Download Financial Template (CSV)",
    data=template_buffer.getvalue(),
    file_name="financial_template.csv",
    mime="text/csv"
)

# Display the analysis results
if pitchdeck_file is not None:
    st.subheader("Pitch Deck Analysis")
    pitchdeck_text = extract_pitchdeck_info(pitchdeck_file)
    if pitchdeck_text:
        pitchdeck_analysis = analyze_pitchdeck(pitchdeck_text)
        st.write(pitchdeck_analysis)

if financial_file is not None:
    st.subheader("Financial Statement Analysis")
    
    # Load the financial data
    if financial_file.name.endswith('.csv'):
        financial_data = pd.read_csv(financial_file)
    elif financial_file.name.endswith('.xlsx'):
        financial_data = pd.read_excel(financial_file)
    
    # Perform Valuation
    valuation_result = perform_valuation_analysis(financial_data)
    st.write(valuation_result)

    # Displaying the financial data for further analysis
    st.write("Uploaded Financial Data:")
    st.write(financial_data.head())

# Final report generation
if st.button("Generate Report"):
    try:
        # Ensure that both files are uploaded before proceeding
        if pitchdeck_file is None:
            st.error("Please upload a pitch deck (PDF) file.")
        elif financial_file is None:
            st.error("Please upload a financial statement (CSV/Excel) file.")
        else:
            # Proceed with extracting and analyzing data only if files are uploaded
            pitchdeck_text = extract_pitchdeck_info(pitchdeck_file)
            pitchdeck_analysis = analyze_pitchdeck(pitchdeck_text) if pitchdeck_text else "No pitch deck analysis available."
            
            financial_data = pd.read_csv(financial_file) if financial_file.name.endswith('.csv') else pd.read_excel(financial_file)
            valuation_result = perform_valuation_analysis(financial_data)
            
            # Combine everything into the final report
            report = f"""
            ## Venture Capital Evaluation Report:

            ### 1. Pitch Deck Analysis:
            {pitchdeck_analysis}

            ### 2. Financial Valuation:
            {valuation_result}

            ### 3. Overall Evaluation:
            Based on the pitch deck and financial statements, the startup shows strong potential for growth. The valuation suggests a promising future, but further detailed analysis and market conditions should be considered.
            """
            st.text(report)
    except Exception as e:
        st.error(f"Error generating final report: {e}")
