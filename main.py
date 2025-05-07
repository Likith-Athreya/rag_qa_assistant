# RAG-Powered Multi-Agent Q&A Assistant (using Hugging Face)

# --- Step 1: Data Ingestion & Chunking ---
import os
import glob
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load and chunk documents
def ingest_documents(path="docs"):
    files = glob.glob(os.path.join(path, "*.txt"))
    docs = []
    for file in files:
        loader = TextLoader(file)
        docs.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return chunks

# --- Step 2: Vector Store & Retrieval ---
def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

def retrieve_top_k(vectordb, query, k=3):
    return vectordb.similarity_search(query, k=k)

# --- Step 3: LLM Integration using Hugging Face (Local Model) ---
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "tiiuae/falcon-rw-1b"  # lightweight and free to run locally

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

llm = HuggingFacePipeline(pipeline=pipe)
qa_chain = load_qa_chain(llm, chain_type="stuff")

def answer_question_with_rag(query, vectordb):
    context = retrieve_top_k(vectordb, query)
    answer = qa_chain.run(input_documents=context, question=query)
    return answer, context

# --- Step 4: Agentic Workflow ---
def is_tool_query(query):
    return any(kw in query.lower() for kw in ["calculate", "compute", "define"])

def use_tool(query):
    if "calculate" in query.lower() or "compute" in query.lower():
        try:
            expression = query.lower().replace("calculate", "").replace("compute", "")
            result = eval(expression)
            return f"The result is: {result}"
        except:
            return "Sorry, I couldn’t compute that."
    elif "define" in query.lower():
        word = query.lower().split("define")[-1].strip()
        return f"Definition of '{word}': [Dummy definition or dictionary API]"
    return "Tool not found."

def agentic_query_handler(query, vectordb):
    log = ""
    if is_tool_query(query):
        log = "Tool path taken."
        response = use_tool(query)
        context = []
    else:
        log = "RAG path taken."
        response, context = answer_question_with_rag(query, vectordb)
    return log, response, context

# --- Step 5: Minimal Streamlit UI ---
import streamlit as st

st.title("RAG Multi-Agent Q&A Assistant (Hugging Face Edition)")
user_query = st.text_input("Ask a question:")

if user_query:
    with st.spinner("Processing..."):
        if "vectordb" not in st.session_state:
            docs = ingest_documents()
            st.session_state.vectordb = build_vector_store(docs)
        log, answer, context = agentic_query_handler(user_query, st.session_state.vectordb)

        st.markdown(f"**Agent Log:** {log}")
        if context:
            st.markdown("**Retrieved Context:**")
            for doc in context:
                st.text(doc.page_content)
        st.markdown(f"**Answer:** {answer}")
