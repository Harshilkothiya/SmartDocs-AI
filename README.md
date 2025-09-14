# 📌SmartDoc-AI

An **AI-powered chatbot** that allows users to **upload documents (PDF, TXT, etc.)** and interact with them through natural language queries. The system uses **embeddings + LLMs** to provide accurate and context-aware answers, and **retains chat history using memory** for more coherent multi-turn conversations.

---

## 🛠️ Tech Stack

### **Frontend**
- HTML, CSS, JavaScript  
- Classic chatbot UI  
- File upload functionality  

### **Backend**
- **Flask** 
- REST API for AI responses  
- Handles document upload and embedding generation  

### **AI / ML**
- Embedding Models:  
  - HuggingFaceEmbeddings - 'sentence-transformers/all-MiniLM-L6-v2'
- LLM: Gemini model 
- **Memory:** ConversationBufferMemory to store chat history

### **Database**
- Vector Database (ChromaDB) for storing embeddings  

---

## 📖 Workflow

1. **User uploads a document**  
2. **System extracts text**  
3. **Text is split into chunks & embedded**  
4. **Embeddings stored in vector DB**  
5. **User asks a query**  
6. **Relevant chunks retrieved using similarity search**  
7. **LLM generates response using context**  
8. **Chatbot retains conversation history using memory**  
9. **Response displayed in chat UI**  

---

## 📊 Example Use Cases
- 📘 Summarize research papers  
- 🏢 Analyze company reports  
- ⚖️ Review legal documents  
- 🏫 Study assistant for students  
- 💼 HR policy & employee handbook Q&A  
- 🔄 Multi-turn Q&A with conversation memory  

---

## 📸 Screenshots
<img width="1920" height="921" alt="image" src="https://github.com/user-attachments/assets/7760013f-45ba-4706-b438-cc845ea4dbbd" />

<img width="1920" height="922" alt="image" src="https://github.com/user-attachments/assets/5004e66f-fa30-4909-b49b-9e43b2d18071" />

---

## 👨‍💻 Authors
- **Harshil Kothiya**  
