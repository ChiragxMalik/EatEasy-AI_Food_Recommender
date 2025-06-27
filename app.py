import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from rag import get_clean_collection_name  # Import the function from rag.py

# Configuration
DB_PATH = "db"  # Match the path from rag.py
COLLECTION_NAME = get_clean_collection_name("data.json")  # Use same collection name as rag.py
SIMILARITY_THRESHOLD = 0.5  # Filter out low-relevance matches

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
    return Ollama(model="gemma3:1b", temperature=0.6)

# Improved Prompt Template
PROMPT_TEMPLATE = """
You are a helpful local restaurant food recommender. The user is looking for: "{query}"

Here are the most relevant dishes from local restaurants, ordered by relevance score:
{results}

Recommend 2-3 of the best matching dishes. For each recommendation:
1. Name the dish and restaurant
2. Explain why it matches their request
3. Mention the price

Keep your response casual and concise. Only recommend dishes from the provided list.
Do not make up any dishes or restaurants. Do not add any disclaimers or ask questions.
"""

def search_dishes(vector_db, query, k=5):
    """
    Search for dishes with relevance scores
    """
    results = vector_db.similarity_search_with_relevance_scores(query, k=k)
    
    matches = []
    for doc, score in results:
        if score < SIMILARITY_THRESHOLD:
            continue
        matches.append({
            "dish": doc.metadata["dish_name"],
            "restaurant": doc.metadata["restaurant"],
            "description": doc.metadata["description"],
            "price": doc.metadata["price"],
            "category": doc.metadata["category"],
            "score": score
        })
    return matches

def generate_response(llm, query, matches):
    if not matches:
        return "I couldn't find any dishes that match your request. Try describing what you're looking for differently!"
    
    context = ""
    for match in matches:
        context += f"- {match['dish']} from {match['restaurant']}: {match['description']} (${match['price']:.2f}, Score: {match['score']:.2f})\n"

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(query=query, results=context)

# Streamlit UI
st.title("🍽️ Local Restaurant Food Recommender")
st.write("Tell me what kind of food you're craving, and I'll recommend dishes from local restaurants!")

query = st.text_input("What are you in the mood for?", "")

if query:
    with st.spinner("Searching for dishes..."):
        db = load_vector_db()
        llm = load_llm()

        matches = search_dishes(db, query)
        response = generate_response(llm, query, matches)

        st.subheader("🤖 Recommendations:")
        st.write(response)
        
        if st.checkbox("Show all matching dishes"):
            st.subheader("All Matching Dishes:")
            for match in matches:
                with st.expander(f"{match['dish']} at {match['restaurant']}"):
                    st.write(f"**Description:** {match['description']}")
                    st.write(f"**Price:** ${match['price']:.2f}")
                    st.write(f"**Category:** {match['category']}")
                    st.write(f"**Relevance Score:** {match['score']:.2f}")
