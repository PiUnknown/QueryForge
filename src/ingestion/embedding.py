from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json


class EmbeddingGenerator:
    """Generates vector embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model
                       Default is all-MiniLM-L6-v2 (384 dimensions, fast)
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}\n")
    
    def generate_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text string"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def generate_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
        
        Returns:
            Array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Add embeddings to chunk dictionaries.
        
        Args:
            chunks: List of chunk dicts with 'text' field
        
        Returns:
            List of chunks with added 'embedding' field
        """
        if not chunks:
            return []
        
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.generate_batch(texts, batch_size=32)
        
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_copy = chunk.copy()
            chunk_copy['embedding'] = embedding.tolist()
            embedded_chunks.append(chunk_copy)
        
        print(f"Generated {len(embedded_chunks)} embeddings\n")
        return embedded_chunks
    
    def save_to_file(self, chunks: List[Dict], output_file: str):
        """Save embedded chunks to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)
        
        print(f"Saved {len(chunks)} embedded chunks to {output_file}")


def process_all_documents():
    """Complete pipeline: load PDFs, chunk, and generate embeddings"""
    from src.ingestion.pdf_loader import PDFLoader
    from src.ingestion.chunking import DocumentChunker
    
    print("="*60)
    print("DOCUMENT PROCESSING PIPELINE")
    print("="*60 + "\n")
    
    # Step 1: Load PDFs
    print("Step 1: Loading PDFs...")
    loader = PDFLoader("data/raw")
    documents = loader.load_all()
    print(f"Loaded {len(documents)} documents\n")
    
    # Step 2: Chunk documents
    print("Step 2: Chunking documents...")
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=100)
    
    all_chunks = []
    for doc in tqdm(documents, desc="Chunking"):
        chunks = chunker.recursive_chunking(
            doc['full_text'],
            doc['metadata']
        )
        all_chunks.extend(chunks)
    
    print(f"Created {len(all_chunks)} chunks\n")
    
    # Step 3: Generate embeddings
    print("Step 3: Generating embeddings...")
    embedder = EmbeddingGenerator()
    embedded_chunks = embedder.embed_chunks(all_chunks)
    
    # Step 4: Save results
    print("\nStep 4: Saving results...")
    output_dir = Path("data/processed/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    embedder.save_to_file(
        embedded_chunks,
        "data/processed/embeddings/chunks_with_embeddings.json"
    )
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Documents processed: {len(documents)}")
    print(f"Total chunks: {len(embedded_chunks)}")
    print(f"Embedding dimension: {embedder.embedding_dim}")
    
    avg_chunk_size = np.mean([len(c['text']) for c in embedded_chunks])
    print(f"Average chunk size: {avg_chunk_size:.0f} characters")
    
    total_size_mb = Path("data/processed/embeddings/chunks_with_embeddings.json").stat().st_size / (1024 * 1024)
    print(f"Output file size: {total_size_mb:.2f} MB")
    
    print("\nPipeline complete\n")


def main():
    """Test embedding generation"""
    print("="*60)
    print("EMBEDDING GENERATION TEST")
    print("="*60 + "\n")
    
    embedder = EmbeddingGenerator()
    
    # Test on sample texts
    test_texts = [
        "Retrieval-augmented generation combines retrieval with language models.",
        "RAG systems improve factual accuracy by grounding responses in retrieved documents.",
        "Embeddings convert text into dense vector representations."
    ]
    
    print("Testing on sample texts...")
    embeddings = embedder.generate_batch(test_texts)
    
    print(f"\nGenerated {len(embeddings)} embeddings")
    print(f"Shape: {embeddings.shape}")
    print(f"Dimension: {embedder.embedding_dim}")
    
    # Test similarity
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("\n" + "="*60)
    print("SEMANTIC SIMILARITY")
    print("="*60 + "\n")
    
    print("Texts:")
    for i, text in enumerate(test_texts):
        print(f"{i+1}. {text}")
    
    print("\nSimilarity scores:")
    for i in range(len(test_texts)):
        for j in range(i+1, len(test_texts)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            print(f"  Text {i+1} vs Text {j+1}: {sim:.3f}")
    
    print("\nTest complete\n")
    print("To process all documents, run: process_all_documents()")


if __name__ == "__main__":
    main()
    
    print("\n" + "="*60)
    response = input("Process all documents? (y/n): ")
    
    if response.lower() == 'y':
        print()
        process_all_documents()