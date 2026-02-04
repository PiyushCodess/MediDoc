# 🩺 MediDoc Q&A (RAG-Based Clinical Document Question Answering)

MediDoc Q&A is an AI-powered **Retrieval-Augmented Generation (RAG)**
system that allows doctors, researchers, and healthcare professionals to
ask natural language questions from **clinical and biomedical PDF
documents** and receive accurate, source-aware answers.

This project demonstrates strong hands-on expertise in **Generative AI,
LLMs, RAG pipelines, and real-world document processing**, making it
highly relevant for AI/ML and GenAI roles.

------------------------------------------------------------------------

## 🚀 Key Features

-   📄 Upload and process clinical or biomedical PDF documents\
-   ✂️ Intelligent document chunking\
-   🧠 Embedding generation and storage using **FAISS vector database**\
-   🔍 Context-aware question answering using **RAG framework**\
-   🧾 Source-cited answers for reliability and transparency\
-   📝 Automatic summarization of medical documents

------------------------------------------------------------------------

## 🧪 Example Questions

-   *What were the side effects of Drug X?*\
-   *Summarize the eligibility criteria of the clinical trial.*\
-   *What was the primary outcome of the study?*

------------------------------------------------------------------------

## 🛠 Tech Stack

-   **Programming Language:** Python\
-   **LLM Framework:** LangChain\
-   **LLM Provider:** OpenAI\
-   **Vector Database:** FAISS\
-   **Backend API:** FastAPI\
-   **Frontend:** Streamlit

------------------------------------------------------------------------

## 🧠 Architecture Overview

1.  Upload PDF documents\
2.  Extract and chunk text\
3.  Generate embeddings using OpenAI\
4.  Store embeddings in FAISS\
5.  Retrieve relevant chunks based on user query\
6.  Generate final answer using LLM with citations

------------------------------------------------------------------------

## ⚙️ Installation & Setup

``` bash
# Clone the repository
git clone https://github.com/your-username/MediDoc-QA.git
cd MediDoc-QA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Run the Application

``` bash
# Start FastAPI backend
uvicorn app.main:app --reload

# Run Streamlit frontend
streamlit run app.py
```

------------------------------------------------------------------------

## 📌 Use Cases

-   Clinical trial analysis\
-   Biomedical research assistance\
-   Healthcare documentation review\
-   Medical knowledge extraction

------------------------------------------------------------------------

## ⭐ Why This Project Matters

-   Demonstrates **end-to-end GenAI application development**
-   Uses **industry-standard RAG architecture**
-   Handles **real-world unstructured medical data**
-   Aligns strongly with **AI Engineer / GenAI Engineer job
    descriptions**

------------------------------------------------------------------------

## 📜 License

This project is for educational and portfolio purposes.

------------------------------------------------------------------------

## 👤 Author

**Piyush Patrikar**\
B.Tech CSE | Aspiring AI Engineer
