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
