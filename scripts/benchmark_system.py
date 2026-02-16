"""Benchmark script to measure actual system performance"""
import time
from pathlib import Path
import json

def benchmark_ingestion():
    """Measure PDF ingestion time"""
    from src.ingestion.pdf_loader import PDFLoader
    
    print("Benchmarking PDF ingestion...")
    start = time.time()
    
    loader = PDFLoader("data/raw")
    documents = loader.load_all()
    
    elapsed = time.time() - start
    
    return {
        'operation': 'Document Ingestion',
        'time_seconds': elapsed,
        'num_documents': len(documents),
        'num_pages': sum(d['metadata']['num_pages'] for d in documents)
    }

def benchmark_embedding():
    """Measure embedding generation time"""
    from src.ingestion.embedding import process_all_documents
    
    print("\nBenchmarking embedding generation...")
    
    # Check if already exists
    embeddings_file = Path("data/processed/embeddings/chunks_with_embeddings.json")
    if embeddings_file.exists():
        print("Embeddings already exist. Delete to re-benchmark.")
        
        # Just measure loading time
        start = time.time()
        with open(embeddings_file, 'r') as f:
            chunks = json.load(f)
        elapsed = time.time() - start
        
        return {
            'operation': 'Embedding Loading',
            'time_seconds': elapsed,
            'num_chunks': len(chunks),
            'note': 'Generation skipped (already exists)'
        }
    else:
        start = time.time()
        process_all_documents()
        elapsed = time.time() - start
        
        return {
            'operation': 'Embedding Generation',
            'time_seconds': elapsed
        }

def benchmark_indexing():
    """Measure vector DB indexing time"""
    from src.retrieval.vector_store import load_and_index
    
    print("\nBenchmarking vector database indexing...")
    start = time.time()
    
    load_and_index()
    
    elapsed = time.time() - start
    
    return {
        'operation': 'Vector DB Indexing',
        'time_seconds': elapsed
    }

def benchmark_retrieval():
    """Measure retrieval speed"""
    from src.retrieval.retriever import RAGRetriever
    
    print("\nBenchmarking retrieval speed...")
    retriever = RAGRetriever(top_k=5)
    
    test_queries = [
        "What is retrieval augmented generation?",
        "How do transformers work?",
        "Explain BERT architecture"
    ]
    
    times = []
    for query in test_queries:
        start = time.time()
        retriever.retrieve(query, top_k=5)
        elapsed = time.time() - start
        times.append(elapsed)
    
    return {
        'operation': 'Query (Retrieval Only)',
        'time_seconds': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'num_queries': len(test_queries)
    }

def benchmark_full_rag():
    """Measure full RAG pipeline speed"""
    from src.generation.llm_chain import RAGChain
    
    print("\nBenchmarking full RAG pipeline...")
    rag = RAGChain(llm_model="llama2", top_k=5)
    
    test_queries = [
        "What is retrieval augmented generation?",
        "How do transformers work?"
    ]
    
    times = []
    for query in test_queries:
        print(f"  Testing: {query[:50]}...")
        start = time.time()
        rag.answer_question(query, top_k=3)
        elapsed = time.time() - start
        times.append(elapsed)
    
    return {
        'operation': 'Query (Full RAG)',
        'time_seconds': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'num_queries': len(test_queries)
    }

def format_time(seconds):
    """Format seconds into readable string"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"

def main():
    print("="*60)
    print("SYSTEM PERFORMANCE BENCHMARK")
    print("="*60)
    print("\nThis will measure actual performance on your system")
    print("Results will be saved to: benchmarks.json\n")
    
    results = []
    
    # Run benchmarks
    try:
        results.append(benchmark_ingestion())
    except Exception as e:
        print(f"Ingestion benchmark failed: {e}")
    
    try:
        results.append(benchmark_embedding())
    except Exception as e:
        print(f"Embedding benchmark failed: {e}")
    
    try:
        results.append(benchmark_indexing())
    except Exception as e:
        print(f"Indexing benchmark failed: {e}")
    
    try:
        results.append(benchmark_retrieval())
    except Exception as e:
        print(f"Retrieval benchmark failed: {e}")
    
    try:
        results.append(benchmark_full_rag())
    except Exception as e:
        print(f"Full RAG benchmark failed: {e}")
    
    # Display results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60 + "\n")
    
    print(f"{'Operation':<30} {'Time':<15} {'Notes'}")
    print("-"*60)
    
    for result in results:
        time_str = format_time(result['time_seconds'])
        notes = result.get('note', '')
        if 'num_documents' in result:
            notes = f"{result['num_documents']} docs, {result['num_pages']} pages"
        elif 'num_chunks' in result:
            notes = f"{result['num_chunks']} chunks"
        elif 'num_queries' in result:
            notes = f"avg of {result['num_queries']} queries"
        
        print(f"{result['operation']:<30} {time_str:<15} {notes}")
    
    # Save to file
    with open('benchmarks.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Results saved to: benchmarks.json")
    print("="*60)
    print("\nUpdate your README with these actual numbers!\n")

if __name__ == "__main__":
    main()