import streamlit as st
from bs4 import BeautifulSoup
import os
import pandas as pd
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_BASE_URL')

def load_stock_symbols():
    try:
        df = pd.read_csv('list_of_stocks.csv')
        # Create a dict of symbol -> company name for display
        return {row['SYMBOL']: f"{row['SYMBOL']} - {row['NAME OF COMPANY']}" 
                for _, row in df.iterrows()}
    except Exception as e:
        st.error(f"Error loading stock symbols: {str(e)}")
        return {}

def fetch_html_content(stock_symbol):
    url = f"https://www.screener.in/company/{stock_symbol}/consolidated"
    
    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in headless mode
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    try:
        # Initialize the driver
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        
        # Wait for key elements to load
        wait = WebDriverWait(driver, 10)
        
        # Wait for peers section to be present and visible
        peers_section = wait.until(
            EC.presence_of_element_located((By.ID, "peers"))
        )
        
        # Additional explicit wait to ensure dynamic content loads
        time.sleep(5)
        
        # Get the page source after waiting
        html_content = driver.page_source
        
        driver.quit()
        return html_content
        
    except Exception as e:
        st.error(f"Error fetching data for {stock_symbol}: {str(e)}")
        if 'driver' in locals():
            driver.quit()
        return None

def extract_financial_info(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    info = {}

    # 1. Company Name
    company_name = soup.find("h1")
    company_name = company_name.get_text(strip=True) if company_name else "Unknown Company"
    info["Company Name"] = company_name

    # 2. Financial Key-Value Ratios
    ratios = soup.find_all("li", class_="flex flex-space-between")
    for ratio in ratios:
        label_tag = ratio.find("span", class_="name")
        value_tag = ratio.find("span", class_="nowrap value")
        if label_tag and value_tag:
            label = label_tag.get_text(strip=True)
            value = value_tag.get_text(" ", strip=True).replace("\n", "")
            info[label] = value

    # 3. Company Profile Details
    profile_div = soup.find("div", class_="company-profile")
    if profile_div:
        about = profile_div.find("div", class_="sub show-more-box about")
        key_points = profile_div.find("div", class_="sub commentary always-show-more-box")

        if about and about.text.strip():
            info["About Company"] = about.text.strip()

        if key_points and key_points.text.strip():
            info["Key Business Highlights"] = key_points.text.strip()

    # 4. Pros and Cons Section
    pros_div = soup.find("div", class_="pros")
    cons_div = soup.find("div", class_="cons")

    if pros_div:
        pros_list = [li.get_text(strip=True) for li in pros_div.find_all("li")]
        if pros_list:
            info["Pros"] = "\n".join([f"- {item}" for item in pros_list])

    if cons_div:
        cons_list = [li.get_text(strip=True) for li in cons_div.find_all("li")]
        if cons_list:
            info["Cons"] = "\n".join([f"- {item}" for item in cons_list])

    # 5. Peer Comparison Table
    peers_section = soup.find("section", id="peers")
    if peers_section:
        peer_table = peers_section.find("table")
        if peer_table:
            rows = peer_table.find_all("tr")
            if len(rows) > 1:
                headers = [th.get_text(strip=True).replace("\n", " ") for th in rows[0].find_all("th")]
                peer_data = []

                for row in rows[1:]:
                    cols = row.find_all("td")
                    if len(cols) == len(headers):
                        row_data = [col.get_text(strip=True) for col in cols]
                        peer_data.append(dict(zip(headers, row_data)))
                    elif len(cols) == len(headers) - 1:
                        row_data = ["-"] + [col.get_text(strip=True) for col in cols]
                        peer_data.append(dict(zip(headers, row_data)))

                peer_summary = "\n".join(
                    [", ".join(f"{k}: {v}" for k, v in row.items()) for row in peer_data]
                )
                info["Peer Comparison"] = peer_summary

    # 6. Profit & Loss Tables
    profit_loss_section = soup.find("section", id="profit-loss")
    if profit_loss_section:
        data_tables = profit_loss_section.find_all("table", class_="data-table")
        pl_data_combined = []
        for table in data_tables:
            thead = table.find("thead")
            tbody = table.find("tbody")
            if not thead or not tbody:
                continue

            header_cells = thead.find_all("th")
            years = [cell.get_text(strip=True) for cell in header_cells[1:]]

            for row in tbody.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if len(cells) < 2:
                    continue
                metric = cells[0].get_text(strip=True)
                values = [cell.get_text(strip=True) for cell in cells[1:]]
                year_value_pairs = [f"{y}: {v}" for y, v in zip(years, values)]
                formatted_row = f"{metric} - " + ", ".join(year_value_pairs)
                pl_data_combined.append(formatted_row)

        if pl_data_combined:
            info["Profit & Loss Summary"] = "\n".join(pl_data_combined)

    # 7. Quarterly Results
    quarters_section = soup.find("section", id="quarters")
    if quarters_section:
        table = quarters_section.find("table", class_="data-table")
        if table:
            thead = table.find("thead")
            tbody = table.find("tbody")
            if thead and tbody:
                header_cells = thead.find_all("th")
                quarters = [cell.get_text(strip=True) for cell in header_cells[1:]]

                quarterly_data = []
                for row in tbody.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    if len(cells) < 2:
                        continue
                    category = cells[0].get_text(strip=True)
                    values = [cell.get_text(strip=True) for cell in cells[1:]]
                    qtr_value_pairs = [f"{q}: {v}" for q, v in zip(quarters, values)]
                    formatted_row = f"{category} - " + ", ".join(qtr_value_pairs)
                    quarterly_data.append(formatted_row)

                if quarterly_data:
                    info["Quarterly Results Summary"] = "\n".join(quarterly_data)

    # 8. Shareholding Pattern
    shareholding_section = soup.find("section", id="shareholding")
    if shareholding_section:
        table = shareholding_section.find("table", class_="data-table")
        if table:
            thead = table.find("thead")
            tbody = table.find("tbody")
            if thead and tbody:
                header_cells = thead.find_all("th")
                periods = [cell.get_text(strip=True) for cell in header_cells[1:]]

                shareholding_data = []
                for row in tbody.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    if len(cells) < 2:
                        continue
                    category = cells[0].get_text(strip=True)
                    values = [cell.get_text(strip=True) for cell in cells[1:]]
                    period_value_pairs = [f"{p}: {v}" for p, v in zip(periods, values)]
                    formatted_row = f"{category} - " + ", ".join(period_value_pairs)
                    shareholding_data.append(formatted_row)

                if shareholding_data:
                    info["Shareholding Pattern"] = "\n".join(shareholding_data)

    # 9. Balance Sheet
    balance_sheet_section = soup.find("section", id="balance-sheet")
    if balance_sheet_section:
        balance_sheet_table = balance_sheet_section.find("table", class_="data-table")
        if balance_sheet_table:
            thead = balance_sheet_table.find("thead")
            tbody = balance_sheet_table.find("tbody")
            if thead and tbody:
                header_cells = thead.find_all("th")
                years = [cell.get_text(strip=True) for cell in header_cells[1:]]

                balance_data = []
                for row in tbody.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    if len(cells) < 2:
                        continue
                    category = cells[0].get_text(strip=True)
                    values = [cell.get_text(strip=True) for cell in cells[1:]]
                    year_value_pairs = [f"{y}: {v}" for y, v in zip(years, values)]
                    formatted_row = f"{category} - " + ", ".join(year_value_pairs)
                    balance_data.append(formatted_row)

                if balance_data:
                    info["Balance Sheet Summary"] = "\n".join(balance_data)

    # 10. Cash Flow Table
    cash_flow_section = soup.find("section", id="cash-flow")
    if cash_flow_section:
        table = cash_flow_section.find("table", class_="data-table")
        if table:
            thead = table.find("thead")
            tbody = table.find("tbody")
            if thead and tbody:
                header_cells = thead.find_all("th")
                years = [cell.get_text(strip=True) for cell in header_cells[1:]]

                cash_flow_data = []
                for row in tbody.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    if len(cells) < 2:
                        continue
                    category = cells[0].get_text(strip=True)
                    values = [cell.get_text(strip=True) for cell in cells[1:]]
                    year_value_pairs = [f"{y}: {v}" for y, v in zip(years, values)]
                    formatted_row = f"{category} - " + ", ".join(year_value_pairs)
                    cash_flow_data.append(formatted_row)

                if cash_flow_data:
                    info["Cash Flow Summary"] = "\n".join(cash_flow_data)

    return info

def build_langchain_prompt(info_dict, question):
    context = "\n".join([f"{k}: {v}" for k, v in info_dict.items()])

    system_template = """
        You are a sophisticated financial analyst AI designed to evaluate stock investment potential based on comprehensive market data. Analyze the following aspects with their respective weightages:

        1. QUARTERLY PERFORMANCE TREND (25% weightage):
        - Revenue Growth: QoQ and YoY comparison
        - EBITDA Margins: Trend and sustainability
        - PAT Growth: Quality of earnings
        - Operating Metrics: Segment-wise performance
        - Volume Growth: Product/Service delivery trends
        - Key Ratio Changes: Working capital, debt metrics

        2. FINANCIAL STATEMENTS ANALYSIS (25% weightage):
        a) Profit & Loss Metrics:
            - Revenue Mix: Segment-wise contribution
            - Gross Margins: Raw material cost impact
            - Operating Margins: Fixed cost efficiency
            - Other Income: Non-core earnings dependency
            - Tax Rate: Effective tax implications

        b) Balance Sheet Health:
            - Debt Profile: Long-term vs Short-term
            - Fixed Assets: Capacity utilization
            - Working Capital: Inventory and receivables
            - Net Worth: Book value trends
            - Contingent Liabilities: Off-balance sheet items

        c) Cash Flow Quality:
            - Operating Cash Flow: Conversion ratio
            - Working Capital Changes: Cash cycle
            - CAPEX Trends: Growth investments
            - Free Cash Flow: Dividend sustainability
            - Financing Activities: Debt servicing

        3. SHAREHOLDING ANALYSIS (20% weightage):
        - Promoter Holding: Changes and pledging
        - FII/DII Holdings: Institutional confidence
        - Public Shareholding: Retail investor interest
        - Block Deals: Major stakeholder changes
        - Employee Stock Options: Management alignment
        - Share Buybacks/Issues: Capital allocation

        4. BUSINESS & MARKET POSITION (15% weightage):
        - Market Share: Industry position
        - Competitive Advantages: Entry barriers
        - Management Quality: Execution track record
        - Corporate Governance: Board composition
        - Future Growth Plans: CAPEX and expansion
        - Industry Cycle Position: Sector outlook

        5. VALUATION & RISK METRICS (15% weightage):
        - Relative Valuation: PE, PB, EV/EBITDA
        - Peer Comparison: Industry benchmarking
        - Risk Factors: Business & market risks
        - Regulatory Environment: Policy impacts
        - Global Factors: External dependencies

        RESPONSE FORMAT:

        INVESTMENT RATING: [Strong Buy / Buy / Hold / Sell / Strong Sell]

        CONFIDENCE SCORE: [0-100%]

        QUARTERLY PERFORMANCE SUMMARY:
        • Revenue Trend: [Last 4-6 quarters analysis]
        • Margin Progression: [Expansion/Contraction with reasons]
        • Key Metrics: [Important operational KPIs]

        FINANCIAL HEALTH INDICATORS:
        • P&L Strength: [Key observations]
        • Balance Sheet Position: [Major changes]
        • Cash Flow Quality: [Operating efficiency]

        SHAREHOLDING INSIGHTS:
        • Promoter Actions: [Recent changes]
        • Institutional Activity: [FII/DII movements]
        • Ownership Quality: [Concentration analysis]

        KEY STRENGTHS:
        • [List top 3-4 positive factors]

        KEY RISKS:
        • [List top 3-4 risk factors]

        VALUATION SUMMARY:
        • Current Valuation: [Undervalued/Fair/Overvalued]
        • Target Price Range: [Lower-Upper band]
        • Investment Horizon: [Short/Medium/Long term]

        PRICE METRICS:
• Current Price: [Current market price]
• 52-week Range: [High/Low analysis]
• Price Trend vs Sector: [Relative performance]

        DECISION RATIONALE:
        [3-4 sentences explaining the final recommendation with specific data points]

        MONITORING POINTS:
        • [Key metrics to track]
        • [Potential trigger events]
        • [Risk mitigation factors]

        Note: 
        1. All trends should be analyzed for minimum 8-12 quarters for pattern recognition
        2. Highlight any unusual patterns or one-time impacts
        3. Flag data inconsistencies or missing information
        4. Consider industry-specific metrics and cycles
        """

    human_template = f"""
    Context:
    {context}

    Question: {question}
    """

    messages = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": human_template}
    ]
    return messages

def ask_with_langchain(info_dict, question):
    llm = ChatOpenAI(
        model_name="(paid) gpt-4o",
        temperature=0.2,
        openai_api_key=openai_api_key,
        base_url=base_url
    )
    messages = build_langchain_prompt(info_dict, question)
    response = llm.invoke(messages)
    return response.content.strip()

def main():
    st.set_page_config(page_title="📊 Screener Stock Analysis", layout="wide")
    
    # Add custom CSS to limit the content width
    st.markdown("""
        <style>
            .main > div {
                max-width: 50%;
                margin: auto;
                padding: 0 1rem;
            }
        </style>
    """, unsafe_allow_html=True)
    #st.title("📈 Screener Stock Analyzer")

    # Load stock symbols
    stock_symbols = load_stock_symbols()
    
    if not stock_symbols:
        st.error("Could not load stock symbols. Please check if list_of_stocks.csv exists.")
        return

    if not openai_api_key:
        st.warning("⚠️ OpenAI API Key not found. Please check your `.env` file.")
        return

    # Create a list of formatted options for the selectbox
    options = list(stock_symbols.values())
    options.insert(0, "")  # Add empty option at the start
    
    # Keep track of previous input length
    if 'prev_length' not in st.session_state:
        st.session_state.prev_length = 0
    
    # Function to clear selection when user starts deleting from end
    def clear_on_last_char():
        current = st.session_state.get('stock_selector', '')
        # If current length is less than previous length, user is deleting
        if len(current) < st.session_state.prev_length:
            st.session_state.stock_selector = ''
        st.session_state.prev_length = len(current)
    
    # Single searchable dropdown with callback
    selected = st.selectbox(
        "Select a stock:",
        options=options,
        key="stock_selector",
        on_change=clear_on_last_char
    )
    
    # Extract symbol from selection
    if selected and selected != "":
        symbol = selected.split(" - ")[0]  # Get symbol part before the dash
        
        # Fetch and process data
        with st.spinner(f"Fetching data for {symbol}..."):
            html_content = fetch_html_content(symbol)
            if html_content:
                info = extract_financial_info(html_content)
                
                # Display company name and basic info
                if "Company Name" in info:
                    st.subheader(info["Company Name"])

                # Create a single column for input with wider width
                col1 = st.container()
                
                # Make the text area take up more width
                st.markdown("""
                <style>
                    .stTextArea textarea {width: 100%;}
                </style>
                """, unsafe_allow_html=True)
                
                with col1:
                    # Question input
                    question = st.text_area(
                        "Ask a question about the company:",
                        height=100,
                        placeholder="e.g., What is the current market cap? How has the quarterly performance been?"
                    )
                    
                    if st.button("Analyze", type="primary"):
                        if question:
                            with st.spinner("Analyzing..."):
                                try:
                                    answer = ask_with_langchain(info, question)
                                    
                                    # Process and display the response
                                    sections = answer.split('\n\n')
                                    
                                    for section in sections:
                                        if section.strip():
                                            if section.startswith('**'):
                                                # Remove extra asterisks after header
                                                header_end = section.find('**', 2)
                                                if header_end != -1:
                                                    header = section[:header_end+2]
                                                    content = section[header_end+2:].strip()
                                                    st.markdown(header)
                                                    if content:
                                                        if '•' in content:
                                                            lines = content.strip().split('\n')
                                                            for line in lines:
                                                                if line.strip():
                                                                    st.markdown(line.strip())
                                                        else:
                                                            st.markdown(content)
                                            
                                            elif 'PROS AND CONS' in section.upper():
                                                lines = section.split('\n')
                                                st.write('**PROS AND CONS**')
                                                for line in lines[1:]:  # Skip the header
                                                    if line.strip().startswith('Pros:'):
                                                        st.write('🟢 ' + line.strip())
                                                    elif line.strip().startswith('Cons:'):
                                                        st.write('🔴 ' + line.strip())
                                                st.write('')  # Add spacing
                                            
                                            else:
                                                # Regular content
                                                st.write(section.strip())
                                                st.write('')  # Add spacing
                                                
                                except Exception as e:
                                    st.error(f"Error generating analysis: {str(e)}")
                        else:
                            st.warning("Please enter a question.")

                # Process data in col2 but don't display raw info
                # with col2:
                #     # Show extracted data in an expandable section
                #     with st.expander("View Raw Data"):
                #         for key, value in info.items():
                #             st.markdown(f"**{key}**")
                #             st.text(value)
                #             st.markdown("---")

if __name__ == "__main__":
    main()
