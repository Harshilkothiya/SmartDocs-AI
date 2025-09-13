from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.vectorstores import Chroma
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv

load_dotenv()


embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

@tool
def summarizer(text:str)->str:
    '''this is function that tack text and summarized it'''
    s_text = llm.invoke(f"summarize this text:\n{text}")
    return s_text



@tool
def query(query: str, text: str)->str:
    '''this function ans the question base on related text from the document'''

    response = llm.invoke(f"ans the following query base on the give text if you don provide the ans then simply return i am not able to ans this query \n{query} and \n {text}")
    return response

@tool
def connect_vectorstore(vector_store):
    '''this function connect vector store as retriver'''
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

@tool
def doc_loader(path:str):
    '''load the file and spilt into the chunck and store into the vector store'''
    if path.endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        # text file
        loader = TextLoader(path)
    
    doc = loader.load()
    
    # split the text 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200

    )

    text_split = splitter.split_documents(doc)

    # store into the database
    vector_store = Chroma.from_documents(
        embedding=embedding,
        documents=text_split
    )

    vector_store.persist()
    return vector_store

if __name__ == '__main__':
    database = doc_loader("cricket.txt")

    # make the agent
    tools = [database, query, connect_vectorstore, summarizer]

    prompt = hub.pull('hwchase17/react')

    agent = create_react_agent(
        llm = llm,
        tools=tools,
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        tools=tools,
        agent=agent,
        verbose=True
    )

    response = agent_executor.invoke({'input':"base on this file can you summrize the morden cricket"})

    print(response)




    