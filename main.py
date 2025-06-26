import os
import uuid
import sqlite3
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.utilities import SQLDatabase
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from google.generativeai import configure, GenerativeModel
import re

# ✅ Configure Gemini safely
configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = GenerativeModel("gemini-1.5-flash")

# ✅ Updated Gemini wrapper with error and null handling
def gemini_llm(prompt, max_new_tokens=256):
    try:
        response = gemini_model.generate_content(prompt)
        text = getattr(response, "text", "").strip()
        return [{"generated_text": text if text else "[Gemini returned no content]"}]
    except Exception as e:
        return [{"generated_text": f"[Gemini error: {str(e)}]"}]

app = FastAPI(title="Gemini RAG Chatbot", version="1.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
CHROMA_PATH = "chroma_store"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

# ✅ Compatible and public embedding model
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

sql_prompt = PromptTemplate(
    input_variables=["columns", "question"],
    template=(
        "You are an expert in SQLite.\n"
        "You're querying a table called 'uploaded_table'.\n"
        "It contains the following columns: {columns}\n"
        "User asked: '{question}'\n"
        "Write a valid SQL SELECT query to answer this. Do not include backticks, markdown, or explanation.\n"
        "Your output should be a plain SQL statement that starts with SELECT."
    )
)

class AskRequest(BaseModel):
    question: str
    db_id: str

@app.get("/")
def home():
    return {"message": "Gemini RAG chatbot is up and running!"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
    with open(filepath, "wb") as f:
        f.write(await file.read())

    df = pd.read_csv(filepath)
    db_path = os.path.join(UPLOAD_DIR, f"{file_id}.db")
    df.to_sql("uploaded_table", sqlite3.connect(db_path), index=False, if_exists="replace")

    docs = [", ".join([f"{col}={row[col]}" for col in df.columns]) for _, row in df.iterrows()]
    vectordb.add_texts(docs, metadatas=[{"source": file_id}] * len(docs))

    return {"db_id": file_id, "message": "File uploaded and indexed!"}

def get_column_names(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(uploaded_table);")
    cols = [row[1] for row in cursor.fetchall()]
    conn.close()
    return cols

@app.post("/ask")
def ask_question(request: AskRequest):
    db_path = os.path.join(UPLOAD_DIR, f"{request.db_id}.db")
    if not os.path.exists(db_path):
        return {"error": "Invalid database ID"}

    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

    try:
        columns = get_column_names(db_path)
        col_string = ", ".join(columns)
        dynamic_prompt = sql_prompt.format(columns=col_string, question=request.question)

        sql_raw = gemini_llm(dynamic_prompt)[0]["generated_text"]
        sql_query = re.sub(r"```(?:sql)?\s*([\s\S]*?)\s*```", r"\1", sql_raw).strip()

        if not re.match(r"(?i)^select\s+", sql_query):
            raise ValueError(f"Generated invalid SQL: {sql_query}")

        result = db.run(sql_query)

    except Exception as e:
        return {"error": str(e), "sql_query": sql_query if 'sql_query' in locals() else None}

    docs = vectordb.similarity_search(request.question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    final_prompt = (
        f"You are a data analyst. A user asked: '{request.question}'.\n"
        f"The SQL query result is:\n{result}\n"
        f"Related information retrieved from the database:\n{context}\n"
        f"Please provide a concise, helpful answer."
    )
    answer = gemini_llm(final_prompt)[0]["generated_text"]

    return {
        "question": request.question,
        "sql_query": sql_query,
        "sql_result": result,
        "rag_context": context,
        "answer": answer
    }
