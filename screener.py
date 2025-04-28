import streamlit as st
st.set_page_config(page_title="Screener Stock Lookup", page_icon="📈")
import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain.text_splitter import TokenTextSplitter

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')

def fetch_all_screener_sections(stock_symbol):
    url = f"https://www.screener.in/company/{stock_symbol}/"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    sections = {}
    
    def extract_table_from_div(div_id, label):
        try:
            section = soup.find("section", {"id": div_id})
            if not section:
                return

            table = section.find("table", {"class": "data-table"}) or section.find("table")
            if not table:
                return

            # Get headers
            headers = []
            header_row = table.find("tr")
            if header_row:
                for th in header_row.find_all("th"):
                    header_text = th.get_text(strip=True)
                    # Skip S.No. column
                    if header_text == "S.No.":
                        continue
                    # Add units if present
                    unit_span = th.find("span")
                    if unit_span:
                        unit = unit_span.get_text(strip=True)
                        if unit and unit not in ["Rs.", "%"]:
                            header_text = f"{header_text} ({unit})"
                    if header_text:
                        headers.append(header_text)

            if not headers:
                return

            # Get data rows
            rows = []
            rows.append(" | ".join(headers))  # Add headers as first row

            # Process data rows
            for row in table.find_all("tr")[1:]:  # Skip header row
                cols = []
                cells = row.find_all("td")[1:] if row.find_all("td") else []  # Skip S.No. column
                
                for cell in cells:
                    # For shareholding pattern, get the category name from the first column
                    if div_id == "shareholding" and cell.get("class") and "text" in cell.get("class"):
                        category = cell.get_text(strip=True)
                        # Remove any numbers in parentheses from category name
                        category = category.split("(")[0].strip()
                        cols.append(category)
                    # For peer comparison, get company name from link
                    elif cell.get("class") and "text" in cell.get("class"):
                        link = cell.find("a")
                        if link:
                            cols.append(link.get_text(strip=True))
                    else:
                        cols.append(cell.get_text(strip=True))

                if len(cols) == len(headers):  # Only add rows that match header count
                    rows.append(" | ".join(cols))

            if len(rows) > 1:  # Only add if we have data rows
                sections[label] = "\n".join(rows)

        except Exception as e:
            pass  # Silently skip any errors

    # Extract company profile information
    company_profile = soup.find("div", {"class": "company-profile"})
    if company_profile:
        # Extract About section
        about_div = company_profile.find("div", {"class": "about"})
        if about_div:
            sections["About Company"] = about_div.get_text(strip=True)

    # Extract pros and cons from analysis section
    analysis_section = soup.find("section", {"id": "analysis"})
    if analysis_section:
        # Extract pros
        pros_div = analysis_section.find("div", {"class": "pros"})
        if pros_div:
            pros_items = [item.get_text(strip=True) for item in pros_div.find_all("li")]
            if pros_items:
                sections["Pros"] = "\n".join(f"• {item}" for item in pros_items)

        # Extract cons
        cons_div = analysis_section.find("div", {"class": "cons"})
        if cons_div:
            cons_items = [item.get_text(strip=True) for item in cons_div.find_all("li")]
            if cons_items:
                sections["Cons"] = "\n".join(f"• {item}" for item in cons_items)

    # Extract company ratios from top-ratios section
    ratios_div = soup.find("ul", {"id": "top-ratios"})
    if ratios_div:
        company_ratios = []  # Changed from dict to list for simpler handling
        for ratio in ratios_div.find_all("li"):
            name = ratio.find("span", {"class": "name"}).get_text(strip=True)
            value = ratio.find("span", {"class": "value"}).get_text(strip=True)
            if name and value:
                company_ratios.append(f"{name}: {value}")
        if company_ratios:
            sections["Company Ratios"] = "\n".join(company_ratios)

    # Extract main tables using parent div ids
    extract_table_from_div("quarters", "Quarterly Results")
    extract_table_from_div("peers", "Peer Comparison")
    extract_table_from_div("profit-loss", "Profit & Loss")
    extract_table_from_div("balance-sheet", "Balance Sheet")
    extract_table_from_div("cash-flow", "Cash Flow")
    extract_table_from_div("ratios", "Ratios")
    extract_table_from_div("shareholding", "Shareholding Pattern")
    
    if not sections:
        return {"Error": f"No main sections found for '{stock_symbol}' on Screener.in."}
    return sections


def create_vector_store(sections):
    # Create documents for each section
    documents = []
    for label, content in sections.items():
        # Split content into smaller chunks for better retrieval
        lines = content.split('\n')
        for i in range(0, len(lines), 5):  # Process 5 lines at a time
            chunk = '\n'.join(lines[i:i+5])
            if chunk.strip():  # Only add non-empty chunks
                doc = Document(
                    page_content=chunk,
                    metadata={"section": label}
                )
                documents.append(doc)
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    model="(paid) text-embedding-3-large"
    )
    
    # Create vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def get_relevant_context(vector_store, query, k=5):
    # Search for relevant documents
    docs = vector_store.similarity_search(query, k=k)
    
    # Group documents by section and maintain order
    sections_dict = {}
    for doc in docs:
        section = doc.metadata["section"]
        if section not in sections_dict:
            sections_dict[section] = []
        sections_dict[section].append(doc.page_content)
    
    # Format context with section headers
    context_parts = []
    for section, contents in sections_dict.items():
        context_parts.append(f"=== {section} ===\n" + "\n".join(contents))
    
    return "\n\n".join(context_parts)

def load_stock_symbols():
    try:
        df = pd.read_csv('list_of_stocks.csv')
        # Create a dict of symbol -> company name for display
        return {row['SYMBOL']: f"{row['SYMBOL']} - {row['NAME OF COMPANY']}" 
                for _, row in df.iterrows()}
    except Exception as e:
        st.error(f"Error loading stock symbols: {str(e)}")
        return {}

def main():
    st.title("🔎 Screener.in Stock Lookup + LLM")
    
    # Load stock symbols
    stock_symbols = load_stock_symbols()
    
    if stock_symbols:
        # Create a list of formatted options for the selectbox
        options = list(stock_symbols.values())
        options.insert(0, "")  # Add empty option at the start
        
        # Single searchable dropdown
        selected = st.selectbox(
            "Search and select a stock:",
            options=options,
            key="stock_selector"
        )
        
        # Extract symbol from selection
        if selected and selected != "":
            stock_query = selected.split(' - ')[0]
        else:
            stock_query = ""
    else:
        # Fallback to simple text input if CSV loading fails
        stock_query = st.text_input("Enter the stock symbol (e.g., GAIL, TCS, INFY):", "")
    
    user_question = st.text_input("Ask a question about this company (e.g., Show latest quarterly profit, List peers, etc.):", "")
    
    if st.button("Search & Analyze", key="search_button") and stock_query.strip():
        with st.spinner(f"Fetching financial data for '{stock_query}' from Screener.in..."):
            sections = fetch_all_screener_sections(stock_query.strip().upper())
        st.markdown("---")

        if user_question:
            with st.spinner("Creating vector database..."):
                vector_store = create_vector_store(sections)
            
            with st.spinner("Finding relevant information..."):
                context = get_relevant_context(vector_store, user_question)
            
            llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                model="(paid) o3-mini",
                temperature=1)  # Reduced temperature for more consistent analysis

            # Investment analysis prompt
            system_prompt = f"""
You are a financial assistant specialized in analyzing stocks for potential investments.
The user is looking for a simple, easy-to-understand explanation of a company's key financials and fundamentals to help decide whether to invest.

You are given structured data about the company, including:

Company Overview

Current Market Price (CMP) and its 52-week High/Low prices

Market Cap, PE Ratio, ROE, ROCE

Net Profit (after tax and expenses)

Balance Sheet Details

Cash Flow Statement

Key Ratios (like Cash Conversion Cycle, ROCE, Working Capital Days)

Shareholding Pattern (Promoters, FIIs, DIIs, Public)

Peer Comparison with competitors

Pros and Cons of the stock (highlighted)

Your tasks are:

Mention the current market price (CMP) along with the 52-week high and low prices, and briefly state whether the stock is currently near its high, low, or mid-range.

Focus mainly on Net Profit trends over time — whether the company's bottom-line profit has increased or decreased after all taxes and expenses.

Avoid discussing sales, operating profit, or other income unless directly relevant to Net Profit behavior.

Comment on the company's asset strength, debt levels, and cash flow trends briefly, especially if they impact future profitability.

Highlight important financial health indicators (like ROE, ROCE) if they support the profit story.

Analyze shareholding patterns: whether promoter holding is stable/increasing, and any trends in FIIs/DIIs.

Compare the company's Net Profit growth and profitability against industry peers if possible (based on available metrics like ROCE, P/E).

Clearly state the Pros and Cons already provided — in very simple, non-technical language.

Keep the language clear, easy to understand, and free from jargon — explain it like you would to a beginner investor.

Keep the explanation brief, clean, and actionable, focusing on Net Profit performance and sustainability.

Important:

If the company’s Net Profit has shown strong and steady growth, highlight it positively.

If Net Profit is volatile or declining, mention it carefully as a risk.

If the company is trading near its 52-week high, mention the positive momentum but advise valuation caution.

If trading near its 52-week low, suggest possible undervaluation but also advise investigating the reason.

If promoter shareholding is stable or increasing, mention it as a positive sign.

Always close with a one-line investment advice summary, like:

“Based on its stable net profit growth and strong fundamentals, this stock appears to be a good opportunity for long-term investment, subject to monitoring for profitability risks.”
Here is the company data:
{context}
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ]
            with st.spinner("Analyzing with LLM..."):
                raw_answer = llm.invoke(messages).content.strip()
            
            # Format the response for better readability
            st.subheader("📊 Investment Analysis")
            
            # Split the answer into sections based on common patterns
            sections = raw_answer.split('\n\n')
            
            for section in sections:
                if not section.strip():
                    continue
                    
                # Try to identify section titles
                if ':' in section:
                    title, content = section.split(':', 1)
                    st.markdown(f"### {title.strip()}")
                    
                    # Format bullet points
                    points = content.split('•')
                    for point in points:
                        if point.strip():
                            st.markdown(f"• {point.strip()}")
                else:
                    # Handle sections without clear titles
                    lines = section.split('\n')
                    for line in lines:
                        if line.strip():
                            if line.strip().endswith(':'):
                                st.markdown(f"### {line.strip()}")
                            else:
                                st.markdown(f"• {line.strip()}")
                
                st.markdown("")
            
            # Add key metrics in a more visual format
            if "CMP" in raw_answer or "Market Cap" in raw_answer:
                st.markdown("### 📈 Key Metrics")
                col1, col2, col3 = st.columns(3)
                
                # Extract and display metrics
                metrics = {
                    "CMP": r"₹\s*([\d,.]+)",
                    "Market Cap": r"Market Cap[:\s]+(₹[\d,.]+ Cr)",
                    "P/E": r"P/?E[:\s]+([\d,.]+)",
                    "ROCE": r"ROCE[:\s]+([\d,.]+)%",
                    "ROE": r"ROE[:\s]+([\d,.]+)%",
                    "Dividend Yield": r"Dividend[\s]+Yield[:\s]+([\d,.]+)%"
                }
                
                import re
                for metric, pattern in metrics.items():
                    match = re.search(pattern, raw_answer)
                    if match:
                        value = match.group(1)
                        if col1.container():
                            col1.metric(metric, value)
                        elif col2.container():
                            col2.metric(metric, value)
                        else:
                            col3.metric(metric, value)
            
            # Add a final summary box if there's a conclusion
            if "Based on" in raw_answer:
                st.markdown("")
                st.info("💡 " + raw_answer.split("Based on")[-1].strip())

if __name__ == "__main__":
    main()