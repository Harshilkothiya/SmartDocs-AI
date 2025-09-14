from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv

load_dotenv()
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__, static_url_path="/Frontend/static")
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

vectorstore = None
agent_executor = None

# tools
@tool
def summarizer(text: str) -> str:
    """Summarize the provided text."""
    s_text = llm.invoke(f"summarize this text:\n{text}").content
    return s_text


@tool
def answer_questio(text: str) -> str:
    """Use this tool to answer questions from PDF text."""
    response = llm.invoke(
        f"Answer the following query based on the given text. "
        f"If you cannot answer, return 'I am not able to answer this query.'\n\n"
        f"\n\nText: {text}"
    ).content
    return response

def connect_vectorstore():
    """Wrap vectorstore as a retriever tool."""
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="Search and return information from the document."
    )
    return retriever_tool

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/aiagent", methods=["post"])
def aiagent():
    global agent_executor
    if agent_executor is None:
        return jsonify({"error": "No document uploaded yet."}), 400
    print(request.data)
    data = request.get_json()
    user_input = data.get("message", "")
    response = agent_executor.invoke({"input": user_input})
    return jsonify({"output": response['output']})


@app.route("/upload", methods=["post"])
def upload():
    print("upload call")
    global vectorstore, agent_executor

    if "file" not in request.files:
        return jsonify({"error": "file not found"}), 400
    else:
        # now we have file so do all 3 task

        # 1 save the file
        file = request.files["file"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # 2 process the file
        if filepath.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        else:
            loader = TextLoader(filepath)

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        text_split = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(embedding=embedding, documents=text_split)
        vectorstore.persist()

        #built the agent
        retriever_tool = connect_vectorstore()
        tools = [retriever_tool, summarizer, answer_questio]
        prompt = hub.pull("hwchase17/react")

        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        print("upload complate")
        return jsonify({"message": "File uploaded and processed successfully!"})
        

if __name__ == "__main__":
    
    app.run(debug=True, port=3000, host="0.0.0.0")

