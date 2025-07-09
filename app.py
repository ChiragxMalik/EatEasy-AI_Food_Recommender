import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Configuration
DB_PATH = "db"
COLLECTION_NAME = "data"

@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

@st.cache_resource
def load_llm():
    return Ollama(model="Gemma3:1b", temperature=0.6) 

# Prompt Template
PROMPT_TEMPLATE = """
You are a helpful food recommender. The user asked: "{query}"

These are the best-matching dishes from the database. Only recommend dishes from this list. Do not invent new dishes or restaurants.

Available options:
{results}

Pick 2–3 of the most relevant dishes from the list. For each one, say:
- the dish name
- the restaurant
- why it fits the user's request

Write casually, but keep it short and focused. No questions, no extra advice, no suggestions outside the list.
"""



def generate_response(llm, query, docs):
    context = ""
    for doc in docs:
        m = doc.metadata
        context += f"- {m['dish_name']} from {m['restaurant']}: {m['description']} (Price: {m['price']})\n"

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(query=query, results=context)

# Streamlit UI
st.title("🍽️ AI Food Recommender with LLM")
st.write("Type what you're craving!")

query = st.text_input("Your craving:", "")

if query:
    db = load_vector_db()
    llm = load_llm()

    results = db.similarity_search(query, k=5)
    response = generate_response(llm, query, results)

    st.subheader("🤖 Recommendation:")
    st.write(response)
