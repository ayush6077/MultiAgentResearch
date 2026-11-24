# ResearchMind · Multi-Agent AI Research System

An autonomous multi-agent research pipeline built with **LangChain**, **Google Gemini**, and **Streamlit**.

ResearchMind coordinates specialized AI agents:
1. **Search Agent**: Queries the web for recent, reliable information via Tavily Search.
2. **Reader Agent**: Scrapes and cleans the top referenced URL for deep context.
3. **Writer Chain**: Synthesizes the gathered intelligence into a comprehensive report.
4. **Critic Chain**: Evaluates the report with an editorial review and rubric score.
5. **Fact-Checking Agent**: Verifies factual claims, statistics, and dates against the underlying sources.

---

## 📦 Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and set your API keys:
```bash
cp .env.example .env
```

`.env` configuration:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-lite-latest
TAVILY_API_KEY=your_tavily_api_key_here
```

---

## 🖥️ Running the Application

### Streamlit Web App
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Command-Line Execution
```bash
python pipeline.py
```

---

## 📂 Project Structure

```
MutiAgentResearch/
├── app.py            # Streamlit interactive application
├── pipeline.py       # Sequential multi-agent pipeline
├── agents.py         # Search, Reader, Writer, Critic, and Fact-Checking agents
├── tools.py          # Tavily search and web scraping tools
├── requirements.txt  # Project dependencies
├── .env              # Environment keys
├── .env.example      # Environment template
└── README.md         # Documentation
```
