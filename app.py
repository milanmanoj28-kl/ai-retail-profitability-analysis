import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Retail Profitability Assistant",
    layout="centered"
)


# ============================================================
# 2. LOAD API KEY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Groq API key was not found.")
    st.stop()

client = Groq(api_key=api_key)


# ============================================================
# 3. LOAD ANALYSIS SUMMARY
# ============================================================

SUMMARY_FILE = BASE_DIR / "llm_summary.json"


@st.cache_data
def load_summary():

    if not SUMMARY_FILE.exists():
        st.error("llm_summary.json was not found.")
        st.stop()

    try:
        with open(SUMMARY_FILE, "r") as file:
            return json.load(file)

    except json.JSONDecodeError:
        st.error("llm_summary.json is not valid JSON.")
        st.stop()


summary = load_summary()


# ============================================================
# 4. GROUNDED LLM FUNCTION
# ============================================================

def ask_groq(question, summary):

    context = json.dumps(
        summary,
        indent=2,
        default=str
    )

    system_prompt = """
You are a retail profitability analytics assistant.

Answer questions using ONLY the supplied retail analysis context.

STRICT RULES:

1. Do not invent numbers or facts.
2. Do not use outside knowledge.
3. If the context does not contain enough information, say:
   "The provided analysis does not contain enough information
   to answer that reliably."
4. Use exact values from the supplied context.
5. Distinguish observed business statistics from model findings.
6. Do not claim causation from correlation.
7. Risk Score is an engineered risk indicator, not actual profit loss.
8. Model predictions are classifications, not guaranteed future outcomes.
9. Keep answers concise and business-focused.
10. Support important conclusions with the relevant data.
"""

    user_prompt = f"""
RETAIL ANALYSIS CONTEXT
=======================

{context}

USER QUESTION
=============

{question}

Answer the question using ONLY the supplied retail analysis.
"""

    try:

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.1,
            max_completion_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as e:

        return (
            "Unable to generate the answer.\n\n"
            f"API error: {str(e)}"
        )


# ============================================================
# 5. USER INTERFACE
# ============================================================

st.title("Retail Profitability Assistant")

st.write(
    "Ask questions about risk, profit leakage, discounts, "
    "categories, segments, and model predictions."
)


# ============================================================
# 6. EXAMPLE QUESTIONS
# ============================================================

with st.expander("Example questions"):

    st.write("• Which category has the highest leakage?")
    st.write("• What's driving the risk in the Furniture category?")
    st.write("• Which segment has the highest profit leakage?")
    st.write("• Is higher discount associated with lower profit margin?")
    st.write("• What are the most important drivers of risk?")
    st.write("• Which category should management prioritize first?")


# ============================================================
# 7. CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# 8. CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your retail data..."
)


if question:

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Analyzing your data..."):

            answer = ask_groq(
                question,
                summary
            )

        st.markdown(answer)


    # Store answer
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })