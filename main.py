from fastapi import FastAPI
from langserve import add_routes

from langsmith import traceable

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

import os
from dotenv import load_dotenv


# =============================================
# ENV + CONFIG
# =============================================
load_dotenv()

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGCHAIN_API"] = os.getenv("LANGSMITH_API")

APIKEY = os.getenv("GROQ_API")
VECTOR_DB_DIR = "vectorDBstore"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

LLM = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=1,
    api_key=APIKEY
)


# =============================================
# TRACED SUB-STEPS
# =============================================
@traceable
def load_vector_db():
    """Load FAISS DB fresh for every request."""
    if not os.path.exists(VECTOR_DB_DIR):
        raise ValueError("Vector DB not found. Please upload a PDF & build DB in Streamlit.")

    return FAISS.load_local(
        VECTOR_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


@traceable
def build_retriever(DB):
    return DB.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
    )


@traceable
def build_prompt():
    template = """
    You are a helpful assistant.

    Strict rules:
    1. Use ONLY the provided context.
    2. If the answer is not in the context, say: "I don't have enough context to answer this."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )


@traceable(run_type="llm")
def run_llm(prompt_chain, query):
    """LLM call should be traced as LLM run."""
    return prompt_chain.invoke(query)


# =============================================
# DYNAMIC RAG RUNNABLE
# =============================================
class DynamicRAG(Runnable):

    @traceable
    def invoke(self, query, config=None):

        # Step 1: load DB
        DB = load_vector_db()

        # Step 2: build retriever
        retriever = build_retriever(DB)

        # Step 3: build prompt
        prompt = build_prompt()

        # Step 4: build pipeline dynamically
        rag_pipeline = (
            RunnableParallel({
                "context": retriever,
                "question": RunnablePassthrough()
            })
            | prompt
            | LLM
            | StrOutputParser()
        )

        # Step 5: traced LLM request
        return run_llm(rag_pipeline, query)


dynamic_rag = DynamicRAG()


# =============================================
# FASTAPI + LANGSERVE
# =============================================
app = FastAPI(title="Dynamic RAG LangServe Server with LangSmith")

add_routes(
    app,
    dynamic_rag,
    path="/rag"
)
