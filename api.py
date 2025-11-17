from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DB_PATH = "db"
COLLECTION_NAME = "data"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize FastAPI
app = FastAPI(title="EatEasy AI Food Recommender API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Request/Response models
class QueryRequest(BaseModel):
    query: str

class RecommendationResponse(BaseModel):
    recommendation: str
    retrieved_dishes: list

# Prompt Template
PROMPT_TEMPLATE = """
You're a friendly local foodie helping someone find great dishes in St. John's. The user said: "{query}"

CRITICAL RULES:
- ONLY recommend dishes from the list below - don't make anything up
- Talk naturally like you're chatting with a friend
- Use first person ("I'd recommend", "You should try", "I love")
- Be casual and conversational
- Only mention what's actually in the descriptions - don't assume or guess
- IMPORTANT: Put each recommendation in its own paragraph with a blank line between them

Here are the dishes I can recommend from:
{results}

Give 2-3 recommendations in a natural, conversational way. For each dish:
- Tell them what it is and where it's from
- Explain why you think they'd like it based on what they asked for
- Mention the price
- Be honest if something doesn't perfectly match but is still worth trying

FORMAT: Write each recommendation as a separate paragraph with a blank line between them. Start each recommendation on a new line.

Talk like you're texting a friend, not writing a formal review. Keep it real and casual!

Response:"""

# Load resources on startup
vector_db = None
llm = None

@app.on_event("startup")
async def startup_event():
    global vector_db, llm
    
    # Load vector database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )
    
    # Load LLM
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
    )
    
    print("✅ Vector database and LLM loaded successfully!")

def generate_response(query: str, docs):
    """Generate recommendation response"""
    if not docs:
        return "I couldn't find any dishes in the database. Please make sure the vector database is properly loaded."
    
    context = ""
    for doc in docs:
        m = doc.metadata
        price_str = f"${m['price']}" if m['price'] else "Price not available"
        context += f"- {m['dish_name']} from {m['restaurant']}: {m['description']} (Price: {price_str})\n"
    
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(query=query, results=context)

# API Routes
@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    return FileResponse("frontend/html/main.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_db": "loaded" if vector_db else "not loaded",
        "llm": "loaded" if llm else "not loaded"
    }

@app.post("/api/recommend", response_model=RecommendationResponse)
async def get_recommendation(request: QueryRequest):
    """Get food recommendations based on user query"""
    try:
        if not vector_db or not llm:
            raise HTTPException(status_code=500, detail="Services not initialized")
        
        if not request.query or request.query.strip() == "":
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Search for relevant dishes
        results = vector_db.similarity_search(request.query, k=8)
        
        if not results:
            raise HTTPException(status_code=404, detail="No dishes found")
        
        # Generate recommendation
        recommendation = generate_response(request.query, results)
        
        # Format retrieved dishes for response
        retrieved_dishes = [
            {
                "dish_name": doc.metadata.get("dish_name", "Unknown"),
                "restaurant": doc.metadata.get("restaurant", "Unknown"),
                "description": doc.metadata.get("description", ""),
                "price": doc.metadata.get("price", 0),
                "category": doc.metadata.get("category", "")
            }
            for doc in results
        ]
        
        return RecommendationResponse(
            recommendation=recommendation,
            retrieved_dishes=retrieved_dishes
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")

@app.get("/api/restaurants")
async def get_restaurants():
    """Get list of all restaurants"""
    try:
        if not vector_db:
            raise HTTPException(status_code=500, detail="Vector database not initialized")
        
        collection = vector_db._collection
        results = collection.get()
        
        # Extract unique restaurants
        restaurants = {}
        for metadata in results['metadatas']:
            restaurant_name = metadata.get('restaurant', 'Unknown')
            if restaurant_name not in restaurants:
                restaurants[restaurant_name] = 0
            restaurants[restaurant_name] += 1
        
        return {
            "total_restaurants": len(restaurants),
            "total_dishes": len(results['ids']),
            "restaurants": [
                {"name": name, "dish_count": count}
                for name, count in sorted(restaurants.items())
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching restaurants: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
