from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.tools.retriever import create_retriever_tool
from langchain.vectorstores import Chroma
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
@tool
def summarizer(text: str) -> str:
    """Summarize the provided text."""
    s_text = llm.invoke(f"summarize this text:\n{text}").content
    return s_text


@tool
def query(text: str) -> str:
    """Answer a question based on the provided context text. """
    response = llm.invoke(
        f"Answer the following query based on the given text. "
        f"If you cannot answer, return 'I am not able to answer this query.'\n\n"
        f"Query: {query}\n\nText: {text}"
    ).content
    return response


# ---------- HELPERS ----------
def doc_loader(path: str):
    """Load file, split into chunks, and return a Chroma vectorstore."""
    if path.endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path)
    
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    text_split = splitter.split_documents(docs)

    vector_store = Chroma.from_documents(
        embedding=embedding,
        documents=text_split
    )
    vector_store.persist()
    return vector_store


def connect_vectorstore(vector_store):
    """Wrap vectorstore as a retriever tool."""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="Search and return information from the document."
    )
    return retriever_tool


database = doc_loader("cricket.txt")

retriever_tool = connect_vectorstore(database)

# make the agent
tools = [retriever_tool, query, summarizer]


prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    tools=tools,
    agent=agent,
    verbose=True
)

response = agent_executor.invoke({'input':'''I have uploaded a document about cricket.  
1. First, search the document to find information about modern cricket.  
2. Then summarize what you found into a short paragraph.  
3. Finally, based on that text, answer this question: "Why do some players wear protective gear in cricket?"
'''})

response