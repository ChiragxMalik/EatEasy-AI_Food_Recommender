from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
import re

# 1. Load menu data using JSONLoader
def load_menu_data(file_path):
    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.restaurants[] | .menu[] | {dish_name, description, price, category, restaurant: .restaurant}',
        text_content=False  # ✅ Because we're returning dicts, not strings
    )
    documents = loader.load()
    return documents

# 2. Prepare documents: move metadata & set text for embedding
def prepare_documents(documents):
    for doc in documents:
        meta = doc.metadata

        # Safely extract fields and set them in metadata
        doc.metadata = {
            "restaurant": meta.get("restaurant", "unknown"),
            "price": meta.get("price", 0),
            "category": meta.get("category", "unknown"),
            "dish_name": meta.get("dish_name", "unknown"),
            "description": meta.get("description", "")
        }

        # This is the actual text used for search
        doc.page_content = f"{doc.metadata['dish_name']}: {doc.metadata['description']}"

    return documents

# 3. Load HuggingFace embedding model
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Clean collection name from filename
def get_clean_collection_name(file_path):
    file_name = os.path.basename(file_path).replace(".json", "")
    clean_name = re.sub(r"[^a-zA-Z0-9._-]", "_", file_name)
    clean_name = re.sub(r"^[^a-zA-Z0-9]+", "", clean_name)
    return clean_name[:50]

# 5. Store in Chroma vector DB
def store_embeddings(documents, embedding_model, collection_name, persist_directory="menu_db"):
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    return vector_db

# 6. Run the whole pipeline
if __name__ == "__main__":
    file_path = "data.json"  # 🔁 Change if your file is in another folder
    persist_directory = "db"
    collection_name = get_clean_collection_name(file_path)

    documents = load_menu_data(file_path)
    enriched_docs = prepare_documents(documents)
    embedding_model = get_embeddings_model()
    vector_db = store_embeddings(enriched_docs, embedding_model, collection_name, persist_directory)

    print(f"✅ Menu vector DB created with {len(enriched_docs)} items.")
    print(f"📁 Collection: {collection_name}")
    print(f"💾 Saved to: {persist_directory}/")
