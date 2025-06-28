import os
import uuid
import duckdb
import pandas as pd
import streamlit as st
import re
import hashlib

from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from google.generativeai import configure, GenerativeModel

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 🔐 Gemini setup
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = GenerativeModel("gemini-1.5-flash")

def gemini_llm(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return getattr(response, "text", "").strip() or "[Gemini returned no content]"
    except Exception as e:
        return f"[Gemini error: {str(e)}]"

# 📁 Setup
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 🔍 Embeddings
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

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
    st.dataframe(df.head())

    db_path = os.path.join(UPLOAD_DIR, f"{file_id}.duckdb")
    conn = duckdb.connect(database=db_path)
    conn.execute("CREATE OR REPLACE TABLE uploaded_table AS SELECT * FROM df")

    # 🧠 FAISS vector store
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
        sql_prompt = PromptTemplate(
            input_variables=["columns", "question"],
            template=(
                "You are an expert in SQL.\n"
                "You're querying a table called 'uploaded_table'.\n"
                "It contains the following columns: {columns}\n"
                "User asked: '{question}'\n"
                "Write a valid SQL SELECT query to answer this. Do not include markdown or explanations.\n"
                "Only return raw SQL starting with SELECT."
            )
        )
        dynamic_prompt = sql_prompt.format(columns=columns, question=question)
        sql_raw = gemini_llm(dynamic_prompt)
        sql_query = re.sub(r"```(?:sql)?\s*([\s\S]*?)\s*```", r"\1", sql_raw).strip()

        if not sql_query.lower().startswith("select"):
            st.warning("⚠️ Gemini did not return a valid SQL query.")
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
                    f"Provide a helpful answer."
                )
                answer = gemini_llm(final_prompt)
                st.success(answer)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
