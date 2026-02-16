from sentence_transformers import SentenceTransformer
from typing import List, Dict
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.vector_store import VectorStore


class RAGRetriever:
    """Handles document retrieval for the RAG pipeline"""
    
    def __init__(self, 
                 collection_name: str = "research_papers",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 top_k: int = 5):
        """
        Initialize retriever with embedding model and vector store.
        
        Args:
            collection_name: Name of ChromaDB collection
            embedding_model: Sentence-transformers model name
            top_k: Default number of chunks to retrieve
        """
        print("Initializing retriever...")
        
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("Embedding model loaded")
        
        self.vector_store = VectorStore(collection_name=collection_name)
        
        self.top_k = top_k
        
        print(f"Retriever ready with {self.vector_store.collection.count()} indexed chunks\n")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve (uses default if None)
        
        Returns:
            List of dicts with text, metadata, and similarity score
        """
        if top_k is None:
            top_k = self.top_k
        
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        retrieved_chunks = []
        for doc, metadata, distance in zip(
            results['documents'],
            results['metadatas'],
            results['distances']
        ):
            # Convert distance to similarity score
            similarity_score = 1 - distance
            
            retrieved_chunks.append({
                'text': doc,
                'metadata': metadata,
                'score': similarity_score
            })
        
        return retrieved_chunks
    
    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into a context string for the LLM.
        
        Args:
            retrieved_chunks: List of retrieved chunk dicts
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk['metadata'].get('filename', 'Unknown')
            text = chunk['text']
            score = chunk['score']
            
            context_parts.append(
                f"[Source {i}: {source} (Relevance: {score:.2f})]\n{text}\n"
            )
        
        return "\n---\n".join(context_parts)


def main():
    """Test the retriever"""
    print("="*60)
    print("RETRIEVER TEST")
    print("="*60 + "\n")
    
    retriever = RAGRetriever(top_k=5)
    
    test_queries = [
        "What is retrieval augmented generation and how does it work?",
        "Explain the transformer architecture",
        "What are the main challenges in RAG systems?",
        "How do you evaluate RAG performance?"
    ]
    
    for query in test_queries:
        print("="*60)
        print(f"Query: {query}")
        print("="*60 + "\n")
        
        chunks = retriever.retrieve(query, top_k=3)
        
        print(f"Retrieved {len(chunks)} relevant chunks:\n")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"[Result {i}] Score: {chunk['score']:.3f}")
            print(f"Source: {chunk['metadata'].get('filename', 'Unknown')}")
            print(f"Preview: {chunk['text'][:250]}...")
            print()
        
        print("-"*60)
        print("FORMATTED CONTEXT:")
        print("-"*60)
        context = retriever.format_context(chunks[:2])
        print(context[:500] + "...\n")
        print("="*60 + "\n")
    
    print("Retriever test complete\n")


if __name__ == "__main__":
    main()