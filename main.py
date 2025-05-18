import streamlit as st
import os
import re
import datetime
import time
import requests
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')

# Initialize LLM with optimal settings
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    model="(paid) gpt-4o-mini",
    temperature=0.3
)

# URLs to process - prioritized and categorized
URLS = {
    "Top Gainers": "https://in.tradingview.com/markets/stocks-india/market-movers-gainers/",
    "Large Cap": "https://in.tradingview.com/markets/stocks-india/market-movers-large-cap/"
    #"Top Losers": "https://in.tradingview.com/markets/stocks-india/market-movers-losers/"
    #"MoneyControl": "https://www.moneycontrol.com/news/business/markets",
    #"LiveMint": "https://www.livemint.com/market/stock-market-news",
    #"CNBC TV18": "https://www.cnbctv18.com/market/",
    #"Financial Express": "https://www.financialexpress.com/market/"
}

def save_html_to_file(html_content, url):
    """Save HTML content to a file with timestamp"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'page_source_{timestamp}.html'
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML content saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error saving HTML: {e}")
        return None

def fetch_last_stock_table_from_url(url: str, max_rows: int = 50):
    response = requests.get(url)
    response.raise_for_status()  # Raise error if fetch fails

    soup = BeautifulSoup(response.text, "html.parser")

    # Get all tables on the page
    tables = soup.find_all("table")
    if not tables:
        raise ValueError("No tables found on the page.")
    
    # Use the last table
    table = tables[-1]

    # Find all rows in tbody
    rows = table.select("tbody tr")

    # Determine URL type and set column mapping
    is_gainers = "gainers" in url.lower()
    is_large_cap = "large-cap" in url.lower()

    # Define column mappings for different URL types
    gainers_columns = [
        "Symbol", "Change %", "Price", "Volume", "Rel Volume", "Market Cap",
        "P/E", "EPS dil (TTM)", "EPS dil growth (TTM YoY)", "Div yield % (TTM)",
        "Sector", "Analyst Rating"
    ]

    large_cap_columns = [
        "Symbol", "Market Cap", "Price", "Change %", "Volume", "Rel Volume",
        "P/E", "EPS dil (TTM)", "EPS dil growth (TTM YoY)", "Div yield % (TTM)",
        "Sector", "Analyst Rating"
    ]

    # Parse each row into a dictionary
    data = []
    for row in rows[:max_rows]:  # Limit to max_rows if specified
        cells = row.find_all("td")
        if not cells:
            continue

        # Get the first cell which contains symbol and company name
        symbol_cell = cells[0]
        
        # Find the symbol (tickerName) and company name (tickerDescription)
        symbol_element = symbol_cell.find('a', class_='tickerName-GrtoTeat')
        company_element = symbol_cell.find('sup', class_='tickerDescription-GrtoTeat')
        
        if symbol_element and company_element:
            symbol = symbol_element.text.strip()
            company_name = company_element.text.strip()
            name = company_name
        else:
            continue  # Skip if we can't find both symbol and company name

        # Initialize stock_data with common fields
        stock_data = {
            "Symbol": symbol,
            "Name": name
        }

        # Add fields based on URL type
        if is_gainers:
            if len(cells) >= len(gainers_columns):
                stock_data.update({
                    "Change %": cells[1].get_text(strip=True),
                    "Price": cells[2].get_text(strip=True),
                    "Volume": cells[3].get_text(strip=True),
                    "Rel Volume": cells[4].get_text(strip=True),
                    "Market Cap": cells[5].get_text(strip=True),
                    "P/E": cells[6].get_text(strip=True),
                    "EPS dil (TTM)": cells[7].get_text(strip=True),
                    "EPS dil growth (TTM YoY)": cells[8].get_text(strip=True),
                    "Div yield % (TTM)": cells[9].get_text(strip=True),
                    "Sector": cells[10].get_text(strip=True),
                    "Analyst Rating": cells[11].get_text(strip=True)
                })
        elif is_large_cap:
            if len(cells) >= len(large_cap_columns):
                stock_data.update({
                    "Market Cap": cells[1].get_text(strip=True),
                    "Price": cells[2].get_text(strip=True),
                    "Change %": cells[3].get_text(strip=True),
                    "Volume": cells[4].get_text(strip=True),
                    "Rel Volume": cells[5].get_text(strip=True),
                    "P/E": cells[6].get_text(strip=True),
                    "EPS dil (TTM)": cells[7].get_text(strip=True),
                    "EPS dil growth (TTM YoY)": cells[8].get_text(strip=True),
                    "Div yield % (TTM)": cells[9].get_text(strip=True),
                    "Sector": cells[10].get_text(strip=True),
                    "Analyst Rating": cells[11].get_text(strip=True)
                })

        data.append(stock_data)

    # Return data in the requested format
    if is_gainers:
        return {"topgainer": data}
    elif is_large_cap:
        return {"largeCap": data}
    return {"data": data}  # fallback for unknown URL types

# This function is now ready to use with any URL that contains multiple tables. 
# It will extract and return data from the last table only.
# Example: fetch_last_stock_table_from_url("https://example.com/stocks")

def text_clean(text):
    """Clean and normalize text"""
    if not text:
        return None
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\n\t\r]+', '\n', text)
    text = re.sub(r'[^\w\s.,%-]+', '', text)
    return text.strip()

def process_large_content(content_list, max_chunk_size=32000):
    """Process large content by splitting it into manageable chunks"""
    # Initialize text splitter with a conservative chunk size (about 1/4 of model's limit)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=1000,
        length_function=len
    )
    
    # Join all content
    full_content = "\n\n".join(content_list)
    
    # Split into chunks
    chunks = text_splitter.split_text(full_content)
    
    # Return first chunk (most relevant content)
    return chunks[0] if chunks else ""

def extract_stock_info(context, query=None):
    """Extract market information based on query type"""
    if not context:
        return None

    # Convert context to proper format if it's a string
    if isinstance(context, str):
        try:
            market_data = json.loads(context)
        except json.JSONDecodeError:
            print("Error: Invalid JSON data in context")
            return None
    else:
        market_data = context
    
    # Determine which category to use based on the query
    target_category = None
    query = query.lower() if query else ''
    
    # Strict category selection
    if 'large cap' in query or 'large-cap' in query:
        target_category = 'LARGE CAP CATEGORY'
    elif 'top gain' in query or 'gainer' in query:
        target_category = 'TOP GAINERS CATEGORY'
    
    # If no specific category is found in query but it contains 'large', default to large cap
    elif 'large' in query:
        target_category = 'LARGE CAP CATEGORY'
    
    # Default to large cap for market cap related queries
    elif 'market cap' in query:
        target_category = 'LARGE CAP CATEGORY'

    # Format and pre-process the market data
    formatted_data = {}
    
    # Only process the target category if specified
    categories_to_process = [target_category] if target_category else market_data.keys()
    
    for category, stocks in market_data.items():
        if category not in categories_to_process:
            continue
            
        formatted_stocks = []
        for stock in stocks:
            # Clean up the price and market cap values
            price = stock.get('Price', '').replace('INRINR', '').strip()
            market_cap = stock.get('Market Cap', '—').replace('INR', '').strip()
            change_pct = stock.get('Change %', '0%')
            
            # Convert change percentage to float for sorting
            try:
                change_value = float(change_pct.strip('%').replace('−', '-'))
            except ValueError:
                change_value = 0.0
            
            # Skip stocks with negative or zero change if query asks for positive change
            if ('positive' in query or 'shows positive change' in query) and change_value <= 0:
                continue
                
            formatted_stock = {
                'Name': stock.get('Stock', ''),
                'Price': f'₹{price}',
                'Change %': change_pct,
                'Change Value': change_value,  # Added for sorting
                'Volume': stock.get('Volume', ''),
                'Market Cap': market_cap,
                'Sector': stock.get('Sector', '').title()
            }
            formatted_stocks.append(formatted_stock)
        
        # Pre-sort stocks by Change Value in descending order
        formatted_stocks.sort(key=lambda x: x['Change Value'], reverse=True)
        formatted_data[category] = formatted_stocks

    # Updated system prompt focused on stock updates with category awareness
    system_prompt = """
    You are an expert stock market analyst. Analyze the provided market data and answer user questions with the following strict guidelines:

    == CRITICAL CATEGORY AND SORTING RULES ==
    1. The data has been pre-filtered to ONLY include stocks from the correct category based on the query.
    2. The stocks are already pre-sorted by Change % in descending order.
    3. YOU MUST USE ALL STOCKS PROVIDED in the data, in the EXACT ORDER given.
    4. DO NOT switch between categories or re-sort the data.
    5. If no stocks are provided, respond with "No stocks found matching the criteria."

    == RESPONSE FORMAT ==
    Number each stock entry and include:
    1. Stock Name
    2. Current Price
    3. Change % (this is the primary sorting field)
    4. Market Cap
    5. Volume
    6. Sector

    == NUMBER FORMATTING ==
    - Use Indian number format with commas
    - Use K for thousands, M for millions, B for billions, T for trillions
    - Always show ₹ symbol for prices
    - Remove any 'INR' or 'INRINR' suffixes
    
    User Query: {query if query else 'List the top 5 large cap stocks by market capitalization'}
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"Analyze this market data:\n{json.dumps(formatted_data, indent=2)}"
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

    # Single chat-style input for any question
    user_query = st.text_input(
        "Ask anything about Indian stock markets, companies, or financial news:",
        placeholder="e.g., Give me a list of top 5 stocks with reasons for their performance"
    )

    # Add analyze button
    analyze_button = st.button("🔍 Analyze Market", type="primary")

    if analyze_button:
        # Create two columns for progress and results
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Processing Sources")
            progress_bar = st.progress(0)
            
            # Process each source
            market_data = {}
            for i, source in enumerate(URLS.keys()):
                st.write(f"📰 Processing {source}...")
                result = fetch_last_stock_table_from_url(URLS[source])
                if result:
                    # Get the stocks list based on URL type
                    stocks = []
                    category = ""
                    if "topgainer" in result:
                        stocks = result["topgainer"]
                        category = "TOP GAINERS CATEGORY"
                        st.write("📈 Top Gainers")
                    elif "largeCap" in result:
                        stocks = result["largeCap"]
                        category = "LARGE CAP CATEGORY"
                        st.write("💰 Large Cap Stocks")
                    else:
                        stocks = result.get("data", [])
                        category = "OTHER CATEGORY"

                    # Format stocks for this category
                    if category:
                        category_stocks = []
                        for stock in stocks:
                            stock_info = {
                                "Stock": stock.get('Name', '').upper(),
                                "Price": f"{stock.get('Price', 'N/A')}INR",
                                "Change %": stock.get('Change %', 'N/A'),
                                "Volume": stock.get('Volume', 'N/A'),
                                "Market Cap": stock.get('Market Cap', '—'),
                                "Sector": stock.get('Sector', 'N/A').lower()
                            }
                            category_stocks.append(stock_info)
                        market_data[category] = category_stocks
                progress_bar.progress((i + 1) / len(URLS))
            
            # Convert market data to string format
            all_content = [json.dumps(market_data, indent=2)]

        with col2:
            if all_content:
                # st.write('LLM Data', all_content)
                st.subheader(f"📈 Analysis")
                # Process content to handle large text
                processed_content = process_large_content(all_content)
                
                # st.write('processed content', processed_content)
                # Get analysis based on the processed content
                analysis = extract_stock_info(processed_content, user_query)
                
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
