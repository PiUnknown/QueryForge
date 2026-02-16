import chromadb
from chromadb.config import Settings
from typing import List, Dict
import json
from pathlib import Path
from tqdm import tqdm


class VectorStore:
    """Manages vector database operations using ChromaDB"""
    
    def __init__(self, persist_directory: str = "data/vector_db", 
                 collection_name: str = "research_papers"):
        """
        Initialize ChromaDB client and collection.
        
        Args:
            persist_directory: Where to store the database
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"Vector store initialized")
        print(f"  Collection: {collection_name}")
        print(f"  Location: {persist_directory}")
        print(f"  Current items: {self.collection.count()}\n")
    
    def add_chunks(self, chunks: List[Dict], batch_size: int = 100):
        """
        Add chunks with embeddings to the vector database.
        
        Args:
            chunks: List of chunk dicts with 'embedding', 'text', and 'metadata'
            batch_size: Number of chunks to add at once
        """
        if not chunks:
            print("No chunks to add")
            return
        
        print(f"Adding {len(chunks)} chunks to vector database...")
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Adding batches"):
            batch = chunks[i:i + batch_size]
            
            ids = [f"chunk_{i + j}" for j in range(len(batch))]
            embeddings = [chunk['embedding'] for chunk in batch]
            documents = [chunk['text'] for chunk in batch]
            metadatas = [chunk['metadata'] for chunk in batch]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        
        print(f"Added {len(chunks)} chunks")
        print(f"Total items in collection: {self.collection.count()}\n")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> Dict:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Vector embedding of the query
            top_k: Number of results to return
        
        Returns:
            Dict with ids, documents, metadatas, and distances
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return {
            'ids': results['ids'][0],
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        count = self.collection.count()
        
        if count > 0:
            sample = self.collection.peek(limit=min(5, count))
            
            return {
                'total_chunks': count,
                'collection_name': self.collection_name,
                'sample_metadata': sample['metadatas'][:3] if sample['metadatas'] else []
            }
        
        return {'total_chunks': 0}
    
    def clear(self):
        """Remove all items from the collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Cleared collection: {self.collection_name}")


def load_and_index():
    """Load embedded chunks and index them in the vector database"""
    print("="*60)
    print("VECTOR DATABASE INDEXING")
    print("="*60 + "\n")
    
    embeddings_file = "data/processed/embeddings/chunks_with_embeddings.json"
    
    if not Path(embeddings_file).exists():
        print(f"Embeddings file not found: {embeddings_file}")
        print("Run embedding.py first to generate embeddings")
        return
    
    print(f"Loading embeddings from: {embeddings_file}")
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks with embeddings\n")
    
    vector_store = VectorStore()
    
    current_count = vector_store.collection.count()
    if current_count > 0:
        print(f"Warning: Collection already has {current_count} items")
        response = input("Clear and re-index? (y/n): ")
        if response.lower() == 'y':
            vector_store.clear()
            print()
    
    vector_store.add_chunks(chunks, batch_size=100)
    
    stats = vector_store.get_stats()
    
    print("="*60)
    print("INDEXING COMPLETE")
    print("="*60)
    print(f"Total chunks indexed: {stats['total_chunks']}")
    print(f"Collection name: {stats['collection_name']}")
    
    print("\nVector database is ready for retrieval\n")
    
    return vector_store


def test_search():
    """Test vector search functionality"""
    print("="*60)
    print("SEARCH TEST")
    print("="*60 + "\n")
    
    vector_store = VectorStore()
    
    if vector_store.collection.count() == 0:
        print("Vector database is empty")
        print("Run load_and_index() first")
        return
    
    from sentence_transformers import SentenceTransformer
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded\n")
    
    test_queries = [
        "What is retrieval augmented generation?",
        "How do transformers work?",
        "What are the benefits of fine-tuning?"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-"*60)
        
        query_embedding = model.encode(query).tolist()
        results = vector_store.search(query_embedding, top_k=3)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'], 
            results['metadatas'], 
            results['distances']
        ), 1):
            print(f"\n[Result {i}] Distance: {distance:.3f}")
            print(f"Source: {metadata.get('filename', 'Unknown')}")
            print(f"Text: {doc[:200]}...")
        
        print("\n" + "="*60 + "\n")
    
    print("Search test complete\n")


def main():
    """Main execution"""
    vector_store = load_and_index()
    
    if vector_store:
        print("\nRunning search test...\n")
        test_search()


if __name__ == "__main__":
    main()