"""
Run the FastAPI server for the frontend
"""
import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting EatEasy AI Food Recommender API")
    print("=" * 60)
    print("\n📍 API will be available at: http://localhost:8000")
    print("📍 Frontend will be available at: http://localhost:8000")
    print("📍 API docs at: http://localhost:8000/docs")
    print("\n⏳ Loading models and vector database...")
    print("=" * 60)
    print()
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
