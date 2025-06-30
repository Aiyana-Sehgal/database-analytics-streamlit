import os
import uuid
import duckdb
import pandas as pd
import streamlit as st
import re

from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from google.generativeai import configure, GenerativeModel

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from transformers import AutoModel, AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings

# 🔐 Gemini Setup
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = GenerativeModel("gemini-1.5-flash")

def gemini_llm(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return getattr(response, "text", "").strip() or "[Gemini returned no content]"
    except Exception as e:
        return f"[Gemini error: {str(e)}]"

def validate_sql_query(query: str, db_path: str) -> tuple[bool, str]:
    try:
        conn = duckdb.connect(database=db_path)
        conn.execute(f"EXPLAIN {query}")
        return True, ""
    except Exception as e:
        return False, str(e)

def generate_valid_sql(question: str, columns: str, db_path: str, max_retries=2) -> str:
    for i in range(max_retries):
        sql_prompt = PromptTemplate(
            input_variables=["columns", "question"],
            template=(
                "You are an expert in DuckDB SQL.\n"
                "You're querying a table called 'uploaded_table'. It has the following columns: {columns}\n\n"
                "Please follow DuckDB syntax rules carefully:\n"
                "- Always wrap column names with double quotes.\n"
                "- When using aggregate functions (like MAX, AVG), do one of the following:\n"
                "  - Use a GROUP BY clause for non-aggregated columns.\n"
                "  - OR use ANY_VALUE() if the exact value is not critical.\n"
                "  - OR use a subquery to fetch the full row with MAX/MIN/etc.\n\n"
                "User question: '{question}'\n"
                "Return a valid SQL SELECT query that starts with SELECT and uses DuckDB-compatible syntax.\n"
                "Only return the raw SQL without commentary or markdown."
                "Wrap all column names in double quotes, especially if they contain spaces, slashes, or underscores."
            )
        )
        prompt = sql_prompt.format(columns=columns, question=question)
        query = gemini_llm(prompt)
        query_clean = re.sub(r"```(?:sql)?\s*([\s\S]*?)\s*```", r"\1", query).strip()

        is_valid, error = validate_sql_query(query_clean, db_path)
        if is_valid:
            return query_clean
        else:
            question = (
                f"The last SQL query failed with error:\n{error}\n"
                f"Please regenerate a correct DuckDB SQL query that answers: {question}"
            )
    return ""

# 📁 File Storage
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ✅ Safe Embedding Loading (no meta tensors)
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-small-v2",
    model_kwargs={"device": "cpu"}
)

# 🎨 Streamlit UI
st.set_page_config(page_title="Gemini SQL RAG Chatbot", layout="centered")
st.title("📊 Gemini SQL RAG Chatbot")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    file_id = str(uuid.uuid4())
    csv_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.read())

    df = pd.read_csv(csv_path)

    # 🧼 Sanitize column names
    original_columns = df.columns.tolist()
    sanitized_columns = [col.replace("/", "_").replace(" ", "_") for col in original_columns]
    df.columns = sanitized_columns

    st.dataframe(df.head())

    db_path = os.path.join(UPLOAD_DIR, f"{file_id}.duckdb")
    conn = duckdb.connect(database=db_path)
    conn.execute("CREATE OR REPLACE TABLE uploaded_table AS SELECT * FROM df")

    # 🧠 FAISS Vector Store
    docs = [
        Document(
            page_content=", ".join([f"{col}={row[col]}" for col in df.columns]),
            metadata={"source": file_id}
        )
        for _, row in df.iterrows()
    ]
    vectordb = FAISS.from_documents(docs, embedding=embeddings)

    conn.close()
    st.success("File uploaded and embedded!")

    question = st.text_input("🔍 Ask a question about your data")

    if st.button("Run") and question:
        columns = ", ".join(df.columns)
        sql_query = generate_valid_sql(question, columns, db_path)

        if not sql_query or not sql_query.lower().startswith("select"):
            st.warning("⚠️ Gemini was unable to generate a valid SQL query.")
        else:
            st.code(sql_query, language="sql")
            try:
                db = SQLDatabase.from_uri(f"duckdb:///{db_path}")
                result = db.run(sql_query)
                st.write("📈 SQL Result:", result)

                docs = vectordb.similarity_search(question, k=3)
                context = "\n".join([doc.page_content for doc in docs])

                final_prompt = (
                    f"You are a data analyst. A user asked: '{question}'.\n"
                    f"The SQL query result is:\n{result}\n"
                    f"Contextual data from the database:\n{context}\n"
                    f"Provide a helpful, clear response."
                )
                answer = gemini_llm(final_prompt)
                st.success(answer)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
