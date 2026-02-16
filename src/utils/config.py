import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for the RAG system"""
    
    # Model configuration
    USE_LOCAL = True
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "llama2"
    
    # Text processing parameters
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 100
    
    # Retrieval settings
    TOP_K = 5
    
    # File paths
    RAW_DATA_PATH = "data/raw"
    PROCESSED_DATA_PATH = "data/processed"
    VECTOR_DB_PATH = "data/vector_db"
    
    # Vector database
    COLLECTION_NAME = "research_papers"
    
    
    
    