"""Compare different chunking strategies empirically"""
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

from src.evaluation.test_questions import get_all_questions


class ChunkingStrategyComparator:
    """Compare retrieval performance across different chunking strategies"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.strategies = ['fixed_size', 'recursive', 'semantic']
        self.results = {}
    
    def create_strategy_collection(self, 
                                   strategy_name: str,
                                   chunks: List[Dict]) -> chromadb.Collection:
        """Create a temporary ChromaDB collection for a strategy"""
        
        # Create temporary client
        client = chromadb.EphemeralClient()
        
        collection = client.create_collection(
            name=f"test_{strategy_name}",
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"  Adding {len(chunks)} chunks to {strategy_name} collection...")
        
        # Add chunks in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            ids = [f"chunk_{i + j}" for j in range(len(batch))]
            embeddings = [chunk['embedding'] for chunk in batch]
            documents = [chunk['text'] for chunk in batch]
            metadatas = [chunk['metadata'] for chunk in batch]
            
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        
        return collection
    
    def evaluate_strategy(self, 
                         collection: chromadb.Collection,
                         test_questions: List[Dict],
                         strategy_name: str) -> Dict:
        """Evaluate a chunking strategy on test questions"""
        
        print(f"\n  Evaluating {strategy_name} strategy...")
        
        precision_at_3 = []
        precision_at_5 = []
        mrr_scores = []
        
        for test_q in tqdm(test_questions, desc=f"  Testing {strategy_name}"):
            question = test_q['question']
            expected_sources = set(test_q['expected_sources'])
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(question).tolist()
            
            # Retrieve top-5
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )
            
            # Extract sources
            retrieved_sources = [
                meta.get('filename', '') 
                for meta in results['metadatas'][0]
            ]
            
            # Calculate precision@3 and precision@5
            top_3_sources = set(retrieved_sources[:3])
            top_5_sources = set(retrieved_sources)
            
            p3 = len(top_3_sources.intersection(expected_sources)) / 3
            p5 = len(top_5_sources.intersection(expected_sources)) / 5
            
            precision_at_3.append(p3)
            precision_at_5.append(p5)
            
            # Calculate MRR
            mrr = 0.0
            for rank, source in enumerate(retrieved_sources, start=1):
                if source in expected_sources:
                    mrr = 1.0 / rank
                    break
            mrr_scores.append(mrr)
        
        return {
            'strategy': strategy_name,
            'mean_precision_at_3': np.mean(precision_at_3),
            'mean_precision_at_5': np.mean(precision_at_5),
            'mean_mrr': np.mean(mrr_scores),
            'std_precision_at_3': np.std(precision_at_3),
            'std_precision_at_5': np.std(precision_at_5),
            'std_mrr': np.std(mrr_scores)
        }
    
    def run_comparison(self):
        """Run full chunking strategy comparison"""
        
        print("="*60)
        print("CHUNKING STRATEGY COMPARISON")
        print("="*60 + "\n")
        
        # Load embeddings for each strategy
        embeddings_dir = Path("data/processed/embeddings")
        
        print("Step 1: Preparing chunks for all strategies\n")
        
        chunks_by_strategy = {}
        
        # Load existing embeddings
        for strategy in self.strategies:
            strategy_file = embeddings_dir / f"chunks_{strategy}.json"
            
            if strategy_file.exists():
                print(f"  Loading {strategy} chunks from file...")
                with open(strategy_file, 'r', encoding='utf-8') as f:
                    chunks_by_strategy[strategy] = json.load(f)
        
        # If no pre-generated chunks, use main file
        if not chunks_by_strategy:
            print("\n  Using main embeddings file (recursive strategy)...")
            main_file = embeddings_dir / "chunks_with_embeddings.json"
            with open(main_file, 'r', encoding='utf-8') as f:
                main_chunks = json.load(f)
            chunks_by_strategy['recursive'] = main_chunks
        
        # Step 2: Evaluate each strategy
        print("\nStep 2: Evaluating each strategy\n")
        
        test_questions = get_all_questions()
        
        comparison_results = []
        
        for strategy_name, chunks in chunks_by_strategy.items():
            print(f"\n[Strategy: {strategy_name.upper()}]")
            print(f"  Total chunks: {len(chunks)}")
            
            # Create collection
            collection = self.create_strategy_collection(strategy_name, chunks)
            
            # Evaluate
            result = self.evaluate_strategy(collection, test_questions, strategy_name)
            comparison_results.append(result)
        
        # Step 3: Display comparison
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60 + "\n")
        
        print(f"{'Strategy':<15} {'P@3':<10} {'P@5':<10} {'MRR':<10} {'Chunks':<10}")
        print("-"*60)
        
        for result in comparison_results:
            strategy = result['strategy']
            num_chunks = len(chunks_by_strategy[strategy])
            
            print(f"{strategy:<15} "
                  f"{result['mean_precision_at_3']:.3f}      "
                  f"{result['mean_precision_at_5']:.3f}      "
                  f"{result['mean_mrr']:.3f}      "
                  f"{num_chunks}")
        
        print("\n" + "="*60)
        
        # Find best strategy
        if comparison_results:
            best_by_p3 = max(comparison_results, key=lambda x: x['mean_precision_at_3'])
            best_by_mrr = max(comparison_results, key=lambda x: x['mean_mrr'])
            
            print("KEY FINDINGS:")
            print("-"*60)
            print(f"Best Precision@3:  {best_by_p3['strategy']} "
                  f"({best_by_p3['mean_precision_at_3']:.3f})")
            print(f"Best MRR:          {best_by_mrr['strategy']} "
                  f"({best_by_mrr['mean_mrr']:.3f})")
        
        print("\n" + "="*60)
        print("COMPARISON COMPLETE")
        print("="*60 + "\n")
        
        print("Use these results to demonstrate:")
        print("   - Empirical comparison of approaches")
        print("   - Data-driven decision making")
        print("   - Understanding of RAG optimization")
        print()
        
        return comparison_results


def main():
    """Run chunking strategy comparison"""
    
    comparator = ChunkingStrategyComparator()
    results = comparator.run_comparison()
    
    # Save results
    output_file = "data/processed/chunking_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}\n")


if __name__ == "__main__":
    main()