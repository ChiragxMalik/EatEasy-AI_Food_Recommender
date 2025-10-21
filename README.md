# 🍽️ AI Local Restaurant Food Recommender

A smart food recommendation system that helps you discover dishes from local restaurants in St. John's, Newfoundland. Built with RAG (Retrieval-Augmented Generation) using local LLMs via Ollama.

## 📸 Screenshot



## 🌟 Features

- **Local LLM Integration**: Uses Ollama with Gemma 3:1b model for privacy-focused recommendations
- **RAG Architecture**: Combines vector search with language models for accurate suggestions
- **Local Restaurant Data**: Features menus from 3 popular St. John's restaurants:
  - The Merchant Tavern
  - Oliver's Restaurant  
  - The Celtic Hearth
- **Smart Search**: Find dishes by cuisine type, ingredients, or dietary preferences
- **Real-time Recommendations**: Get instant suggestions with prices and descriptions
- **No Hallucination**: Only recommends actual dishes from the restaurant menus

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Ollama (Gemma 3:1b, Can be switched)
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Data Processing**: LangChain

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Gemma 3:1b model pulled in Ollama

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ChiragxMalik/EatEasy-AI_Food_Recommender.git
cd EatEasy-AI_Food_Recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Pull the Ollama model:
```bash
ollama pull gemma3:1b
```

4. Create the vector database:
```bash
python rag.py
```

5. Run the application:
```bash
streamlit run app.py
```

## 📋 Usage

1. Open your browser to `http://localhost:8501`
2. Type what you're craving (e.g., "anything green", "something crispy", "spicy food")
3. Get personalized recommendations from local St. John's restaurants
4. View detailed dish descriptions, prices, and restaurant information

## 🏪 Featured Restaurants

- **The Merchant Tavern**: Contemporary Canadian cuisine with local ingredients
- **Oliver's Restaurant**: Classic comfort food and seafood specialties  
- **The Celtic Hearth**: Traditional pub fare with Maritime influences

## 🔧 Configuration

- **LLM Model**: Change model in `app.py` (line 23)
- **Database Path**: Modify `DB_PATH` in `app.py` 
- **Restaurant Data**: Update `data.json` to add more restaurants

