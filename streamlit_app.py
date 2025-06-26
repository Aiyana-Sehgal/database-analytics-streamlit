import os
import uuid
import sqlite3
import pandas as pd
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from google.generativeai import configure, GenerativeModel
import re

# 🔐 Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = GenerativeModel("gemini-1.5-flash")

def gemini_llm(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        text = getattr(response, "text", "").strip()
        return text if text else "[Gemini returned no content]"
    except Exception as e:
        return f"[Gemini error: {str(e)}]"

# ✅ Set up directories
UPLOAD_DIR = "uploads"
CHROMA_PATH = "chroma_store"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

st.set_page_config(page_title="Gemini CSV SQL Assistant", layout="centered")
st.title("📊 Gemini SQL RAG Chatbot")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    file_id = str(uuid.uuid4())
    csv_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.read())

    df = pd.read_csv(csv_path)
    st.dataframe(df.head())

    db_path = os.path.join(UPLOAD_DIR, f"{file_id}.db")
    conn = sqlite3.connect(db_path)
    df.to_sql("uploaded_table", conn, index=False, if_exists="replace")

    docs = [", ".join([f"{col}={row[col]}" for col in df.columns]) for _, row in df.iterrows()]
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    vectordb.add_texts(docs, metadatas=[{"source": file_id}] * len(docs))

    conn.close()
    st.success("File uploaded and embedded!")

    question = st.text_input("🔍 Ask a question about your data")

    if st.button("Run") and question:
        columns = ", ".join(df.columns)
        sql_prompt = PromptTemplate(
            input_variables=["columns", "question"],
            template=(
                "You are an expert in SQLite.\n"
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

        st.code(sql_query, language="sql")

        try:
            db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
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
