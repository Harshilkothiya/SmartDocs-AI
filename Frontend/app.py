from dotenv import load_dotenv
load_dotenv()

import os
from flask import Flask, render_template, request, jsonify

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# ---------------- APP ----------------
app = Flask(__name__, static_url_path="/Frontend/static")
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------------- MODELS ----------------
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Gemini 2.5 Flash is valid in your 2026 context
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Global for demo purposes only (Note: Not production safe)
agent_executor = None

# ---------------- TOOLS ----------------
@tool
def summarizer(text: str) -> str:
    """Summarize the provided text."""
    return llm.invoke(f"Summarize the following text:\n\n{text}").content

# REMOVED: answer_question tool (Redundant and causes errors)

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/aiagent", methods=["POST"])
def aiagent():
    global agent_executor

    if agent_executor is None:
        return jsonify({"error": "No document uploaded yet."}), 400

    data = request.get_json()
    user_input = data.get("message", "").strip() # Remove extra spaces

    # ---------------------------------------------------------
    # 1. HANDLE GREETINGS DIRECTLY (Saves AI cost & Time)
    # ---------------------------------------------------------
    greetings = ["hi", "hello", "hey", "hola", "greetings"]
    
    # Check if the user input (lowercase) is exactly one of the greetings
    if user_input.lower() in greetings:
        return jsonify({
            "output": "Hi! How are you? How may I help you with your document today?"
        })

    # ---------------------------------------------------------
    # 2. IF NOT A GREETING, LET THE AGENT HANDLE IT
    # ---------------------------------------------------------
    try:
        response = agent_executor.invoke({"input": user_input})
        return jsonify({"output": response["output"]})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"output": "I encountered an error processing that request."})

@app.route("/upload", methods=["POST"])
def upload():
    global agent_executor

    if "file" not in request.files:
        return jsonify({"error": "File not found"}), 400

    file = request.files["file"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # ---- Load document ----
    loader = PyPDFLoader(filepath) if filepath.endswith(".pdf") else TextLoader(filepath)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Use in-memory Chroma to avoid locking issues during development
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding
        # persist_directory removed for stability
    )

    # ---- Build Agent ----
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    retriever_tool = create_retriever_tool(
        retriever,
        name="document_search",
        description="Search for specific information in the uploaded document to answer user questions."
    )

    # Only include tools the ReAct agent can handle easily
    tools = [retriever_tool, summarizer]

    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True # Good practice to add this
    )

    return jsonify({"message": "File uploaded and processed successfully!"})

if __name__ == "__main__":
    app.run(debug=True, port=3000, host="0.0.0.0")