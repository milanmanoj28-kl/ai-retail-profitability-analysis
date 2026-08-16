# AI Retail Profitability Analysis

An end-to-end retail intelligence pipeline that goes beyond dashboards and static reporting. This project combines transparent risk engineering, machine learning classification, and an LLM-powered reasoning layer to answer a question most retail analytics stops short of: not just *what happened*, but *where should the business act first*.

Built on the Sample Superstore transaction dataset, the pipeline moves through five distinct stages — cleaning, risk scoring, predictive classification, aggregation, and AI-generated recommendations — culminating in a conversational interface where anyone on a business team can ask plain-English questions and get answers grounded strictly in real data.

---

## Why This Project Exists

Most retail analytics projects stop at visualization: a dashboard that shows profit dropped in a category, without explaining why, without flagging risk before it materializes, and without translating findings into a prioritized action plan. This project was built to close that gap.

The guiding question throughout was simple: **can an analytics pipeline not only diagnose profit leakage, but predict it, explain it in business language, and recommend what to do about it — all while remaining fully auditable and free of hallucinated numbers?**

---

## Core Business Questions Answered

- Where is profit leakage concentrated across categories, regions, and segments?
- How strongly is discounting associated with declining profit margins?
- Can a machine learning model flag high-risk transactions before they fully play out?
- Can an LLM convert dense tabular evidence into recommendations a manager could act on today?
- Can users interrogate the analysis conversationally, without needing to write a single line of SQL or Python themselves?

---

## Pipeline Architecture

Raw Transaction Data
|
Data Cleaning and Validation
|
Explainable Risk Score Engineering
|
Random Forest Risk Classification
|
Aggregated Business Summaries
|
Groq LLM Insight and Recommendation Generation
|
Streamlit Conversational Interface

Every stage feeds the next, but each is also independently interpretable — the risk score can be explained without the model, the model can be explained without the LLM, and the LLM's output can always be traced back to a specific number in the underlying data.

---

## Stage 1: Data Cleaning

The raw dataset (9,994 transactions across 13 columns) was inspected for structure, missing values, and duplication before any transformation was applied. Numeric columns were coerced to proper types, repeated categorical fields were converted to category dtype for efficiency, and 17 exact duplicate rows were removed, producing a clean base of 9,977 transactions. The cleaning process was deliberately conservative — inspect first, transform only what's necessary, and validate the result before moving forward.

---

## Stage 2: Explainable Risk Score

Rather than relying on an opaque model to flag risk, this project engineers a fully transparent, rule-based Risk Score, built so that every number in it can be defended in a single sentence.

The score combines three components:

**Margin Risk (40 percent weight)** — measures how far a transaction's profit margin falls below a healthy 20 percent benchmark, scaling to maximum risk as margin approaches zero or turns negative.

**Discount Risk (40 percent weight)** — measures how far a transaction's discount exceeds a 10 percent acceptable baseline, scaling to maximum risk at a 50 percent discount.

**Quantity Risk (20 percent weight)** — treated as an exposure multiplier, since a risky transaction with a higher quantity carries greater financial consequence.

Risk Score = 100 x (0.40 x Margin Risk + 0.40 x Discount Risk + 0.20 x Quantity Risk)

Margin and discount are weighted equally and heavily because they are the most direct drivers of profitability. Quantity is weighted lower because it amplifies risk rather than causing it directly. Every threshold in the formula (10 percent, 50 percent, 20 percent) is a stated business assumption, not a statistically derived cutoff — and the project is explicit about that distinction rather than disguising it as discovered truth.

Transactions are then bucketed into Low, Medium, and High risk levels for downstream classification and reporting.

---

## Stage 3: Predictive Risk Classification

The engineered Risk Score is converted into a binary target — High Risk versus Not High Risk — and a Random Forest classifier (300 trees, balanced class weighting) is trained to predict that label using only information that would realistically be available at the moment a transaction occurs: discount, category, sub-category, region, segment, ship mode, quantity, and sales.

Critically, the score itself and any directly derived variables (along with profit and profit margin, which are typically only known after the fact) are excluded from the feature set to avoid the model trivially reconstructing its own target. This project treats that leakage risk seriously rather than glossing over it — the model is honestly framed as a classification demonstration, with a stated caveat that because the target was itself engineered from business rules, the model partially relearns those same rules rather than discovering an entirely independent pattern.

The model is evaluated on accuracy, precision, recall, and a full confusion matrix, with particular attention to recall — in a profit-leakage screening context, a missed high-risk transaction is a more costly error than a false alarm.

---

## Stage 4: Aggregated Business Summaries

Rather than exposing an LLM to thousands of raw rows, the project compresses findings into compact, structured summaries: risk distribution by category and sub-category, profit leakage by region and segment, the relationship between discount bands and profit margin, the highest-risk categories, and the classification model's own performance metrics and feature importances.

These summaries are collected into a single structured JSON file, which becomes the sole source of truth for everything the LLM is allowed to say. This design choice keeps the AI layer inexpensive to run, easy to audit, and impossible to hallucinate beyond — the model literally cannot invent a number that isn't already sitting in the JSON in front of it.

---

## Stage 5: AI-Generated Recommendations

Using Groq's LLaMA 3.3 model, the project turns aggregated evidence into ranked, business-ready recommendations. The prompting design enforces strict grounding rules: use only the supplied data, never invent figures or causes, never claim causation from correlation alone, treat the risk score as a screening signal rather than a confirmed loss, and treat model predictions as classifications rather than guaranteed outcomes.

The output follows a consistent structure: an executive recommendation, a ranked list of up to five specific actions, the evidence supporting each one, and an explicit statement of data limitations. The goal is to move the conversation from "what happened" to "what should management prioritize first" — while keeping every claim traceable back to a specific figure in the underlying analysis.

---

## Stage 6: RAG-lite Conversational Interface

The final layer is a Streamlit chat application where users can ask natural-language questions such as "which category has the highest profit leakage" or "what is driving the risk in the Furniture category" and receive answers grounded entirely in the aggregated JSON.

This is intentionally built as a lightweight retrieval pattern rather than a full vector-database RAG system. Because the aggregated evidence is already compact, the entire context can be passed directly into each Groq request without embeddings or similarity search — a pragmatic choice that keeps the system simple, fast, and easy to reason about at this data scale, while still following the core grounded-context principle that defines RAG.

---

## Key Finding

Across the three product categories, Furniture carries both the highest total profit leakage and the highest average risk score, driven primarily by deeper-than-typical discounting relative to its margin structure — a pattern the pipeline surfaces automatically through the aggregation and confirms through the classifier's own feature importance, where discount emerges as the single strongest predictive signal.

---

## Tech Stack

Python, Pandas, NumPy, Scikit-learn (Random Forest), Groq API (LLaMA 3.3), Streamlit, python-dotenv

---

## Project Structure

| File | Purpose |
|---|---|
| SampleSuperstore.csv | Original raw transaction dataset |
| SampleSuperstore_cleaned.csv | Cleaned transaction dataset |
| SampleSuperstore_risk_scored.csv | Dataset enriched with engineered risk fields |
| llm_summary.json | Compact aggregated evidence passed to the LLM |
| app.py | Streamlit chat application and grounding logic |
| requirements.txt | Python dependencies |

---

## Honest Limitations

This project is built to be defensible, not oversold. A few limitations are worth stating plainly rather than hiding:

The classification target is derived from an engineered scoring formula, so the model is partially relearning a rule that was designed rather than discovering an entirely independent future outcome — a stronger version of this project would validate against genuinely held-out, independently observed future losses.

The risk thresholds used throughout (10 percent and 50 percent discount, 20 percent margin, and so on) are business assumptions chosen for interpretability, not values derived from statistical optimization or sensitivity testing.

Feature importance from the Random Forest indicates which inputs the model relied on most heavily — it does not, on its own, establish that any single factor causes profit loss.

---

## Running the Project

Install dependencies:

pip install -r requirements.txt

Create a .env file with your Groq API key:

GROQ_API_KEY=your_key_here

Launch the interface:

streamlit run app.py
