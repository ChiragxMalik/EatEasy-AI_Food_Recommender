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
    return Ollama(model="Gemma3:1b", temperature=0.1) 

# Improved Prompt Template
PROMPT_TEMPLATE = """
You are a helpful restaurant menu assistant. The user asked: "{query}"

CRITICAL INSTRUCTIONS:
- You MUST ONLY use information from the dishes listed below
- DO NOT create, invent, or hallucinate any dishes, restaurants, prices, or descriptions
- Base your explanations ONLY on the actual ingredients/descriptions provided
- NEVER make assumptions about dishes that aren't supported by their descriptions

Here are the ONLY available dishes you can recommend:
{results}

Based ONLY on the dishes listed above, recommend 2-3 dishes that best match the user's query. For each recommendation, provide:
- Exact dish name (as listed above)
- Restaurant name (as listed above)  
- Brief explanation based ONLY on the ingredients/description provided
- Price (as listed above)

Only recommend dishes where the description actually supports your reasoning. If no dishes closely match, say "I don't see any dishes that perfectly match your request, but here are some similar options from our available menus:"

Response:"""

def generate_response(llm, query, docs):
    if not docs:
        return "I couldn't find any dishes in the database. Please make sure the vector database is properly loaded."
    
    context = ""
    for doc in docs:
        m = doc.metadata
        price_str = f"${m['price']}" if m['price'] else "Price not available"
        context += f"- {m['dish_name']} from {m['restaurant']}: {m['description']} (Price: {price_str})\n"
    
    # Debug: Log the context to see what the model is working with
    print("Context passed to LLM:\n", context)
    
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(query=query, results=context)

# Streamlit UI
st.title("🍽️ AI Food Recommender with LLM")
st.write("Type what you're craving!")

query = st.text_input("Your craving:", "")

if query:
    try:
        db = load_vector_db()
        llm = load_llm()

        # Increase k to get more relevant results
        results = db.similarity_search(query, k=8)
        
        if not results:
            st.error("No dishes found in the database. Please run rag.py first to populate the database.")
        else:
            response = generate_response(llm, query, results)
            
            st.subheader("🤖 Recommendation:")
            st.write(response)
            
            # Debug section (optional - can be removed in production)
            with st.expander("Debug: Retrieved Documents"):
                for i, doc in enumerate(results):
                    st.write(f"**{i+1}. {doc.metadata['dish_name']}** from {doc.metadata['restaurant']}")
                    st.write(f"Description: {doc.metadata['description']}")
                    st.write(f"Price: ${doc.metadata['price']}")
                    st.write("---")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Make sure you've run rag.py first to create the vector database.")