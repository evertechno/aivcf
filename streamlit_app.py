import streamlit as st
import google.generativeai as genai
import pandas as pd
import PyPDF2
import io

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("AI-Powered Pitch Deck and Financial Analysis")
st.write("Upload the pitch deck and financial statements, and get a robust analysis with valuation insights and scoring.")

# File uploader for pitch deck and financial statements
uploaded_pitch_deck = st.file_uploader("Upload the Pitch Deck (PDF)", type=["pdf"])
uploaded_financials = st.file_uploader("Upload the Financial Statements (CSV, Excel)", type=["csv", "xlsx"])

# Button to download the template
def create_financial_template():
    """Creates a sample financial template for download."""
    # Example financial data template in Excel format (you can adjust this structure as needed)
    data = {
        "Year": [2021, 2022, 2023],
        "Revenue": [0, 0, 0],  # Placeholder values
        "Cost of Goods Sold": [0, 0, 0],
        "Gross Profit": [0, 0, 0],
        "Operating Expenses": [0, 0, 0],
        "Net Income": [0, 0, 0],
        "Free Cash Flow": [0, 0, 0]
    }

    df = pd.DataFrame(data)
    return df

# Download the financial statement template
st.subheader("Download Financial Statement Template")
template_df = create_financial_template()
csv_template = template_df.to_csv(index=False)

# Provide a button to download the CSV template
st.download_button("Download Template (CSV)", csv_template, "financial_statement_template.csv", "text/csv")

# Extract text from PDF
def extract_pdf_text(file):
    """Extracts text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract data from Excel or CSV
def extract_financial_data(file):
    """Extracts data from an Excel or CSV file."""
    if file.type == "application/vnd.ms-excel" or file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return pd.read_excel(file)
    elif file.type == "text/csv":
        return pd.read_csv(file)

# Process the uploaded files
pitch_deck_text = ""
financial_data = None

if uploaded_pitch_deck:
    if uploaded_pitch_deck.type == "application/pdf":
        pitch_deck_text = extract_pdf_text(uploaded_pitch_deck)

if uploaded_financials:
    financial_data = extract_financial_data(uploaded_financials)

# Function to analyze pitch deck and financials using AI
def analyze_pitch_deck(pitch_deck_text):
    """Uses Google Generative AI to analyze the pitch deck."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Please analyze the following pitch deck and provide insights, key strengths, weaknesses, and opportunities. {pitch_deck_text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error in pitch deck analysis: {e}")
        return None

def analyze_financials(financial_data):
    """Perform financial analysis and valuation."""
    try:
        # Example: Calculating basic financial ratios or growth rate
        financial_data['Growth Rate'] = financial_data['Revenue'].pct_change() * 100
        latest_growth_rate = financial_data['Growth Rate'].iloc[-1]
        return f"Latest revenue growth rate: {latest_growth_rate:.2f}%"
    except Exception as e:
        st.error(f"Error in financial analysis: {e}")
        return None

def valuation_analysis(financial_data):
    """Implement valuation models like DCF, Comparable Companies."""
    try:
        # Placeholder for a simple valuation analysis (DCF, comparable methods, etc.)
        # Assuming we have free cash flow and discount rate
        free_cash_flow = financial_data['Free Cash Flow'].mean()
        discount_rate = 0.1  # Example discount rate
        terminal_growth_rate = 0.02  # Example terminal growth rate
        valuation = free_cash_flow / (discount_rate - terminal_growth_rate)  # Simple DCF formula
        return f"Valuation based on DCF model: ${valuation:,.2f}"
    except Exception as e:
        st.error(f"Error in valuation analysis: {e}")
        return None

# Trigger analysis if files are uploaded
if uploaded_pitch_deck and uploaded_financials:
    st.write("Analyzing the Pitch Deck and Financials...")

    # Analyze pitch deck
    pitch_deck_analysis = analyze_pitch_deck(pitch_deck_text)
    if pitch_deck_analysis:
        st.subheader("Pitch Deck Analysis")
        st.write(pitch_deck_analysis)

    # Analyze financials
    financial_analysis = analyze_financials(financial_data)
    if financial_analysis:
        st.subheader("Financial Analysis")
        st.write(financial_analysis)

    # Perform valuation analysis
    valuation_result = valuation_analysis(financial_data)
    if valuation_result:
        st.subheader("Valuation Analysis")
        st.write(valuation_result)

    # Create downloadable report
    report_data = {
        "Pitch Deck Analysis": pitch_deck_analysis,
        "Financial Analysis": financial_analysis,
        "Valuation Analysis": valuation_result
    }

    # Convert report to a downloadable file (CSV for simplicity)
    report_df = pd.DataFrame(list(report_data.items()), columns=["Section", "Analysis"])
    csv_report = report_df.to_csv(index=False)
    st.download_button("Download Analysis Report", csv_report, "pitch_deck_analysis_report.csv", "text/csv")
else:
    st.write("Please upload both the pitch deck and financial statements to get started.")
