import streamlit as st
st.set_page_config(page_title="Screener Stock Lookup", page_icon="📈")
import requests
from bs4 import BeautifulSoup
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv

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


def main():
    st.title("🔎 Screener.in Stock Lookup + LLM")
    stock_query = st.text_input("Enter the stock symbol (e.g., GAIL, TCS, INFY):", "")
    user_question = st.text_input("Ask a question about this company (e.g., Show latest quarterly profit, List peers, etc.):", "")
    if st.button("Search & Analyze", key="search_button") and stock_query.strip():
        with st.spinner(f"Fetching financial data for '{stock_query}' from Screener.in..."):
            sections = fetch_all_screener_sections(stock_query.strip().upper())
        st.markdown("---")

        if user_question:
            llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                model="(paid) o3-mini",
                temperature=1)
            
            # Define specialized prompts for different types of analysis
            shareholding_prompt = """
When analyzing shareholding patterns:
1. For each category (Promoters, FIIs, DIIs, etc.), compare the latest quarter (rightmost) with the previous quarters
2. Calculate the trend (increasing, decreasing, or stable)
3. Format your response as follows:
   - Latest holding (Mar 2025): X%
   - Year ago (Mar 2024): Y%
   - Change: Show if increased/decreased and by how much
4. Highlight significant changes:
   - If any category shows a consistent uptrend/downtrend over multiple quarters
   - If any category has made a significant change (>2% change)
"""

            pnl_prompt = """
When analyzing Profit & Loss statements:
1. Summarize key financial trends in simple, non-technical language
2. Highlight major changes:
   - Sharp rises or falls in sales
   - Significant profit changes
   - Important margin variations
3. Compare important figures across years
4. Focus on these key metrics:
   - Sales growth trend
   - Operating profit trend and OPM% changes
   - Net profit trend
   - EPS trend
   - Dividend payout pattern
5. Be concise and clear - explain like you're talking to a beginner investor
6. Provide actionable insights when possible
"""

            # Determine which prompt to use based on the question
            question_lower = user_question.lower()
            if any(term in question_lower for term in ["shareholding", "promoter", "fii", "dii", "stake", "holding"]):
                analysis_prompt = shareholding_prompt
            elif any(term in question_lower for term in ["profit", "loss", "pnl", "margin", "sales", "revenue", "eps", "dividend"]):
                analysis_prompt = pnl_prompt
            else:
                analysis_prompt = "Provide a clear and concise analysis based on the available data."

            # Concatenate all sections as context
            context = "\n\n".join([f"=== {label} ===\n{content}" for label, content in sections.items()])

            # Construct the final system prompt
            system_prompt = f"""
You are a financial analyst expert. Use the following company data to answer the user's question as accurately as possible.

{analysis_prompt}

Here is the company data:
{context}
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ]
            with st.spinner("Analyzing with LLM..."):
                answer = llm.invoke(messages).content.strip()
            st.markdown("---")
            st.subheader("LLM Answer")
            st.write(answer)

if __name__ == "__main__":
    main()