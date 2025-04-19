# Indian Stock Market Analysis

A real-time stock market analysis tool that aggregates and analyzes news from multiple Indian financial news sources.

## Features

- Real-time market data from multiple sources
- AI-powered analysis of market trends
- Interactive Q&A about market conditions
- Automatic report generation
- User-friendly web interface

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env`:
   ```
   OPENAI_API_KEY=your_api_key
   OPENAI_BASE_URL=your_base_url
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run market_ui.py
```

## Deployment

This app can be deployed on Streamlit Cloud:

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add your environment variables in the Streamlit Cloud dashboard
5. Deploy!
