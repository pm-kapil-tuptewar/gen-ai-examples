import streamlit as st
import os
import re
import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import tempfile
import pickle

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')

# Initialize LLM with optimal settings
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    # model="(paid) gpt-4o-mini",
    # temperature=0.3,
    model="(paid) o3-mini",
    temperature=1,
    request_timeout=60
)

# Embedding model for FAISS
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    model="(paid) text-embedding-3-large"
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

    # No sidebar, always use all URLs

    # Single chat-style input for any question
    user_query = st.text_input(
        "Ask anything about Indian stock markets, companies, or financial news:",
        placeholder="e.g., Give me a list of top 5 stocks with reasons for their performance"
    )

    # Add analyze button
    analyze_button = st.button("🔍 Analyze Market", type="primary")

    if analyze_button:
        # Always proceed, all URLs are used
        # Create two columns for progress and results
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Processing Sources")
            progress_bar = st.progress(0)
            
            # Process each re-source
            all_content = []
            for i, source in enumerate(URLS.keys()):
                st.write(f"📰 Processing {source}...")
                content = process_url(URLS[source])
                if content:
                    all_content.append(content)
                progress_bar.progress((i + 1) / len(URLS))

        with col2:
            if all_content:
                st.subheader(f"📈 Analysis")
                # --- FAISS Embedding & Retrieval Logic ---
                # 1. Split all_content into chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = []
                for i, content in enumerate(all_content):
                    # Each content is string, wrap as Document
                    docs.extend(splitter.create_documents([content]))
                # 2. Use a temp directory to cache the FAISS index for this session
                faiss_path = os.path.join(tempfile.gettempdir(), "faiss_index")
                if os.path.exists(faiss_path):
                    vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
                else:
                    vectorstore = FAISS.from_documents(docs, embeddings)
                    vectorstore.save_local(faiss_path)
                # 3. Retrieve relevant chunks for user_query
                relevant_docs = vectorstore.similarity_search(user_query, k=5)
                context = "\n\n".join(doc.page_content for doc in relevant_docs)
                print('context', len(context))
                # 4. Get analysis based on only relevant context
                analysis = extract_stock_info(context, user_query)
                
                if analysis:
                    # Add query context and timestamp
                    st.info(f"Query: {user_query}")
                    st.caption(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Display analysis in a clean format
                    st.markdown("---")
                    st.markdown(analysis)
                    
                    # Show sources
                    st.markdown("---")
                    st.caption(f"Sources: {', '.join(URLS.keys())}")
                else:
                    st.warning("⚠️ No relevant information found for your query.")
            else:
                st.error("❌ Could not fetch market data from any source.")

if __name__ == "__main__":
    main()
