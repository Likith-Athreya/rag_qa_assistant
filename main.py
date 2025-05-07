import os
import glob
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from numexpr import evaluate
from PyDictionary import PyDictionary

# --- Config ---
MODEL_NAME = "distilgpt2"  # Lightweight model for local use
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Helper Functions ---
def load_documents():
    """Load and split documents from the 'docs' folder."""
    try:
        files = glob.glob(os.path.join("docs", "*.txt"))
        if not files:
            st.warning("No documents found in the 'docs' folder!")
            return None

        docs = []
        for file in files:
            loader = TextLoader(file)
            docs.extend(loader.load())

        splitter = CharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        return splitter.split_documents(docs)
    except Exception as e:
        st.error(f"⚠️ Failed to load documents: {e}")
        return None

def initialize_vector_db(chunks):
    """Initialize FAISS vector database."""
    try:
        embeddings = HuggingFaceEmbeddings()
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        st.error(f"⚠️ Failed to create vector DB: {e}")
        return None

def initialize_llm():
    """Load Hugging Face LLM."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"⚠️ Failed to load LLM: {e}")
        return None

def calculate(expression):
    """Safe calculation using numexpr."""
    try:
        return f"🔢 Result: {evaluate(expression)}"
    except:
        return "❌ Calculation failed. Please check your input."

def define_word(word):
    """Fetch word definition using PyDictionary."""
    try:
        meanings = PyDictionary().meaning(word)
        if meanings:
            definition = "\n".join([f"📖 {k}: {', '.join(v)}" for k, v in meanings.items()])
            return definition
        else:
            return f"❌ No definition found for '{word}'."
    except:
        return "❌ Dictionary service unavailable."

def retrieve_context(vectordb, query, k=3):
    """Retrieve relevant text chunks."""
    try:
        return vectordb.similarity_search(query, k=k)
    except Exception as e:
        st.error(f"⚠️ Retrieval failed: {e}")
        return []

def generate_rag_answer(llm, query, context):
    """Generate answer using RAG."""
    try:
        context_str = "\n".join([doc.page_content for doc in context])
        prompt = f"""
        Question: {query}
        Context: {context_str}
        Answer concisely:
        """
        return llm(prompt)
    except Exception as e:
        return f"❌ LLM Error: {e}"

# --- Streamlit App ---
def main():
    st.title("🤖 RAG-Powered Q&A Assistant")
    st.markdown("Ask a question or try: *'Calculate 5*5'*, *'Define AI'*")

    # Initialize components
    if "vectordb" not in st.session_state:
        chunks = load_documents()
        if chunks:
            st.session_state.vectordb = initialize_vector_db(chunks)
        st.session_state.llm = initialize_llm()

    # User input
    query = st.text_input("Your question:", placeholder="Type here...")

    if query:
        with st.spinner("🔍 Thinking..."):
            # Agent decision: Tool or RAG?
            if any(kw in query.lower() for kw in ["calculate", "compute", "define"]):
                st.session_state.agent_path = "🛠️ Used Tool"
                if "calculate" in query.lower() or "compute" in query.lower():
                    expr = query.lower().replace("calculate", "").replace("compute", "").strip()
                    response = calculate(expr)
                else:
                    word = query.lower().split("define")[-1].strip()
                    response = define_word(word)
                context = []
            else:
                st.session_state.agent_path = "📚 Used RAG"
                context = retrieve_context(st.session_state.vectordb, query)
                response = generate_rag_answer(st.session_state.llm, query, context)

            # Display results
            st.subheader("Agent Path")
            st.code(st.session_state.agent_path)

            if context:
                st.subheader("Retrieved Context")
                for i, doc in enumerate(context, 1):
                    st.text_area(f"Context {i}", doc.page_content, height=100)

            st.subheader("Answer")
            st.write(response)

if __name__ == "__main__":
    main()