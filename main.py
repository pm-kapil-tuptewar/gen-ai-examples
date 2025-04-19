import streamlit as st
import os
import re
import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')

# Initialize LLM with optimal settings
llm = ChatOpenAI(
    openai_api_key="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",
    model="gpt-4o-mini",
    temperature=0.3,
    request_timeout=60
)

# URLs to process - prioritized and categorized
URLS = {
    "Economic Times": "https://economictimes.indiatimes.com/markets/stocks/news",
    "MoneyControl": "https://www.moneycontrol.com/news/business/markets",
    "LiveMint": "https://www.livemint.com/market/stock-market-news",
    "CNBC TV18": "https://www.cnbctv18.com/market/",
    "Financial Express": "https://www.financialexpress.com/market/"
}

def process_url(url):
    """Process a single URL and extract market information"""
    try:
        with st.spinner(f"Processing {url.split('/')[-2]}..."):
            loader = WebBaseLoader([url])
            docs = loader.load()
            content = "\n\n".join([doc.page_content for doc in docs])
            return text_clean(content)
    except Exception as e:
        st.error(f"Error loading {url}: {e}")
        return None

def text_clean(text):
    """Clean and normalize text"""
    if not text:
        return None
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\n\t\r]+', '\n', text)
    text = re.sub(r'[^\w\s.,%-]+', '', text)
    return text.strip()

def extract_stock_info(context, query=None):
    """Extract market information based on query type"""
    if not context:
        return None

    # Determine query type and set appropriate system prompt
    if query:
        query_lower = query.lower()
        if 'announcement' in query_lower or 'news' in query_lower:
            system_prompt = """You are a financial data analyst. Extract and analyze company announcements and news:\n                            - Focus on corporate actions, regulatory filings, and important updates\n                            - Include dates and specific details\n                            - Categorize by type (e.g., Earnings, M&A, Dividends, etc.)\n                            - Highlight potential market impact\n                            Format the response in clear, categorized points."""
        elif 'result' in query_lower or 'earning' in query_lower:
            system_prompt = """You are a financial data analyst. Extract and analyze company results:\n                            - Include key financial metrics (Revenue, Profit, etc.)\n                            - Show YoY and QoQ comparisons\n                            - Highlight important management comments\n                            - Include analyst expectations vs actuals\n                            Format with clear numbers and percentages."""
        else:
            system_prompt = f"""You are a financial data analyst. Answer the following market query:\n                            {query}\n                            \n                            - Provide specific numbers and facts\n                            - Include relevant company names and sectors\n                            - Add market context where applicable\n                            Format the response in clear, numbered points."""
    else:
        system_prompt = """You are a financial data analyst. Extract stock information in this format:\n                            1. Company Name: ₹Price (±Change%)\n                            - Only include entries with specific numbers\n                            - Sort by percentage change (highest to lowest)\n                            - Include market indices (Nifty, Sensex) at the top\n                            - Maximum 15 most significant stock movements"""
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"Analyze this market information:\n{context}"
        }
    ]
    
    try:
        with st.spinner("Analyzing market data..."):
            response = llm.invoke(messages)
            content = response.content.strip()
            return content if content else None
    except Exception as e:
        st.error(f"Error in analysis: {e}")
        return None

def main():
    st.set_page_config(page_title="Market Analysis", page_icon="📈", layout="wide")
    
    st.title("📊 Indian Stock Market Analysis")
    st.markdown("---")

    # Sidebar for source selection
    st.sidebar.title("Settings")
    selected_sources = st.sidebar.multiselect(
        "Select News Sources",
        list(URLS.keys()),
        default=list(URLS.keys())[:3]
    )

    # Query type selection
    query_type = st.radio(
        "Select the type of information you need:",
        ["Stock Prices", "Company Announcements", "Financial Results", "Custom Query"],
        horizontal=True
    )

    # Example queries based on type
    example_queries = {
        "Stock Prices": [
            "What are the top gaining stocks today?",
            "Show me stocks with highest trading volume",
            "List IT sector stock movements"
        ],
        "Company Announcements": [
            "Show recent company announcements",
            "List latest dividend announcements",
            "Show merger and acquisition news"
        ],
        "Financial Results": [
            "Show latest quarterly results",
            "Companies that beat earnings expectations",
            "IT companies Q4 results"
        ],
        "Custom Query": [
            "Type your own market-related query"
        ]
    }

    # Show example queries
    if query_type != "Custom Query":
        example = st.selectbox(
            "Select an example query or type your own:",
            example_queries[query_type]
        )
        user_query = st.text_input("Customize your query:", value=example)
    else:
        user_query = st.text_input(
            "Enter your query:",
            placeholder="e.g., Which sectors are performing well today?"
        )

    # Add analyze button
    analyze_button = st.button("🔍 Analyze Market", type="primary")

    if analyze_button:
        if not selected_sources:
            st.warning("Please select at least one news source.")
            return

        # Create two columns for progress and results
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Processing Sources")
            progress_bar = st.progress(0)
            
            # Process each selected source
            all_content = []
            for i, source in enumerate(selected_sources):
                st.write(f"📰 Processing {source}...")
                content = process_url(URLS[source])
                if content:
                    all_content.append(content)
                progress_bar.progress((i + 1) / len(selected_sources))

        with col2:
            if all_content:
                st.subheader(f"📈 {query_type} Analysis")
                combined_content = "\n\n".join(all_content)
                
                # Get analysis based on user query
                analysis = extract_stock_info(combined_content, user_query)
                
                if analysis:
                    # Add query context and timestamp
                    st.info(f"Query: {user_query}")
                    st.caption(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Display analysis in a clean format
                    st.markdown("---")
                    st.markdown(analysis)
                    
                    # Show sources
                    st.markdown("---")
                    st.caption(f"Sources: {', '.join(selected_sources)}")
                else:
                    st.warning("⚠️ No relevant information found for your query.")
            else:
                st.error("❌ Could not fetch market data from any source.")

if __name__ == "__main__":
    main()
