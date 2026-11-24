import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url

load_dotenv()

# model setup (Gemini)
model_name = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0,
    api_key=api_key if api_key else "missing-key"
)


# 1st agent: Search Agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )


# 2nd agent: Reader Agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )


# 3rd: Writer Chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# 4th: Critic Chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()


# 5th: Fact-Checking Agent (with Source Verification)
fact_checker_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert fact-checking and source verification agent. You verify factual assertions, statistics, dates, and claims strictly against the research gathered."),
    ("human", """Perform a thorough fact-check and source verification on the research report below.

Topic: {topic}

Report:
{report}

Research Gathered (Ground Truth):
{research}

Respond in this exact structured format:

Factuality Score: X/100
Verification Confidence: High / Moderate / Low

Summary:
- Total Claims Checked: X
- Verified: X
- Partially Verified: X
- Unverified/Disputed: X

Claim-by-Claim Verification:
1. Claim: ...
   Status: [VERIFIED / PARTIALLY VERIFIED / UNVERIFIED]
   Source/URL: ...
   Evidence: ...

Source Verification & Audit:
- Source Validity: [Check if URLs match the gathered research]
- Discrepancies/Biases: [Any unverified claims or 'None detected']

Verdict & Recommendations:
- ..."""),
])

fact_checker_chain = fact_checker_prompt | llm | StrOutputParser()
