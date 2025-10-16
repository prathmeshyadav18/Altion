# **Altion — AI-Powered Portfolio Rebalancer**

An intelligent AI system that helps investors optimize and rebalance their portfolios based on risk tolerance, asset allocation, and AI-generated financial reasoning — powered by **LangChain** and **Groq LLM**.

---

## **Overview**

**Altion** is an AI-driven portfolio rebalancing assistant that leverages **Large Language Models (LLMs)** to provide personalized, explainable investment insights.

The system enables users to:

* Upload their stock portfolios
* Select their **risk profile** (Low / Medium / High)
* Instantly receive a **rebalanced investment strategy** with clear explanations and visual analytics

This bridges the gap between raw financial data and intelligent decision-making, offering a balance between automation and transparency.

---

## **Problem Statement**

Most retail investors lack access to personalized portfolio advisory tools.
Existing platforms display performance metrics but rarely explain the reasoning behind investment changes.

**Altion** addresses this by integrating:

* Real-time data analysis
* Risk-sensitive AI reasoning
* Transparent, educational investment recommendations

Its goal is to democratize financial intelligence and empower every investor with accessible, explainable guidance.

---

## **Tech Stack**

| Layer                      | Tools & Frameworks          |
| -------------------------- | --------------------------- |
| **Language**               | Python 3.12                 |
| **AI Framework**           | LangChain 0.3               |
| **LLM Provider**           | Groq API (Llama3 / Mixtral) |
| **Frontend / UI**          | Streamlit 1.36              |
| **Data Processing**        | Pandas, OpenPyXL            |
| **Visualization**          | Plotly                      |
| **Environment Management** | Python-dotenv               |

---

## **System Architecture**

```
User (Streamlit UI)
     │
     ▼
Portfolio Parser (pandas + openpyxl)
     │
     ▼
Prompt Builder (LangChain)
     │
     ▼
Groq LLM (Llama3 / Mixtral)
     │
     ▼
AI Response + Visualization (Plotly + Streamlit)
```

### **Key Components**

1. **Frontend Layer** – Streamlit interface for uploading portfolio files and visualizing results.
2. **Data Layer** – Cleans, validates, and structures Excel-based portfolio data.
3. **Prompt Builder** – Crafts structured prompts for the Groq LLM based on user risk profile.
4. **AI Reasoning Engine** – Uses Groq’s Llama3/Mixtral models for portfolio rebalancing and explanation generation.
5. **Visualization Layer** – Displays before-and-after allocation charts and rebalancing summaries.

---

## **Core Features**

* Portfolio upload support for Excel and CSV formats
* Risk-based rebalancing tailored to user preferences
* Explainable AI outputs describing the reasoning behind each recommendation
* Interactive comparison of current vs. suggested allocations
* Modular backend for easy integration with future data APIs

---

## **Conceptual Workflow**

1. **Data Input** – User uploads a portfolio file.
2. **Preprocessing** – Data is parsed and structured using Pandas and OpenPyXL.
3. **AI Analysis** – LangChain constructs a contextual prompt and queries the Groq LLM.
4. **Response Generation** – The model produces a rebalanced plan with reasoning.
5. **Visualization** – Streamlit presents the AI-generated insights using interactive charts.

---

## **Impact**

* Processes and analyzes portfolios in under 3 seconds using Groq acceleration
* Handles up to 50+ assets efficiently
* Generates transparent and explainable insights, enhancing financial understanding
* Keeps user data private, as all processing occurs locally

---

## **Future Enhancements**

* Integration with live NSE/BSE stock feeds
* Market sentiment analysis from financial news sources
* Real-time performance tracking and analytics dashboard
* Cloud deployment using Streamlit Cloud or Hugging Face Spaces
* Automated email or message-based portfolio updates

---
