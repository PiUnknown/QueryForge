from typing import List, Dict
import numpy as np
from collections import Counter
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.retriever import RAGRetriever
from src.evaluation.test_questions import get_all_questions


class RAGEvaluator:
    """Evaluates RAG system performance using various metrics"""
    
    def __init__(self, retriever: RAGRetriever):
        """
        Initialize evaluator with a retriever instance.
        
        Args:
            retriever: RAGRetriever instance to evaluate
        """
        self.retriever = retriever
    
    def calculate_precision(self, 
                           question: str, 
                           expected_sources: List[str],
                           top_k: int = 5) -> Dict:
        """
        Calculate precision and recall for retrieval.
        
        Precision = (relevant documents retrieved) / (total documents retrieved)
        Recall = (relevant documents retrieved) / (total relevant documents)
        
        Args:
            question: Query string
            expected_sources: List of relevant document filenames
            top_k: Number of documents to retrieve
        
        Returns:
            Dict with precision, recall, and F1 scores
        """
        chunks = self.retriever.retrieve(question, top_k=top_k)
        
        retrieved_sources = set([
            chunk['metadata'].get('filename', '') 
            for chunk in chunks
        ])
        
        expected_sources_set = set(expected_sources)
        relevant_retrieved = retrieved_sources.intersection(expected_sources_set)
        
        precision = len(relevant_retrieved) / len(retrieved_sources) if retrieved_sources else 0
        recall = len(relevant_retrieved) / len(expected_sources_set) if expected_sources_set else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'retrieved_sources': list(retrieved_sources),
            'expected_sources': expected_sources,
            'relevant_retrieved': list(relevant_retrieved),
            'top_k': top_k
        }
    
    def calculate_mrr(self, 
                     question: str,
                     expected_sources: List[str],
                     top_k: int = 10) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        MRR = 1 / rank of first relevant document
        
        Args:
            question: Query string
            expected_sources: List of relevant document filenames
            top_k: Number of documents to retrieve
        
        Returns:
            MRR score (0 if no relevant doc found)
        """
        chunks = self.retriever.retrieve(question, top_k=top_k)
        
        expected_sources_set = set(expected_sources)
        
        for rank, chunk in enumerate(chunks, start=1):
            source = chunk['metadata'].get('filename', '')
            if source in expected_sources_set:
                return 1.0 / rank
        
        return 0.0
    
    def get_relevance_scores(self, question: str, top_k: int = 5) -> Dict:
        """
        Get statistics on relevance scores for retrieved chunks.
        
        Args:
            question: Query string
            top_k: Number of chunks to retrieve
        
        Returns:
            Dict with score statistics
        """
        chunks = self.retriever.retrieve(question, top_k=top_k)
        
        scores = [chunk['score'] for chunk in chunks]
        
        return {
            'scores': scores,
            'mean_score': np.mean(scores) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'std_score': np.std(scores) if scores else 0
        }
    
    def run_full_evaluation(self, test_questions: List[Dict] = None) -> Dict:
        """
        Run complete evaluation on a test set.
        
        Args:
            test_questions: List of test question dicts (uses default if None)
        
        Returns:
            Dict with summary metrics and detailed results
        """
        if test_questions is None:
            test_questions = get_all_questions()
        
        print("="*60)
        print(f"RUNNING EVALUATION ON {len(test_questions)} QUESTIONS")
        print("="*60 + "\n")
        
        results = {
            'precision_at_3': [],
            'precision_at_5': [],
            'recall_at_3': [],
            'recall_at_5': [],
            'f1_at_3': [],
            'f1_at_5': [],
            'mrr': [],
            'mean_scores': [],
            'per_question_results': []
        }
        
        for i, test_q in enumerate(test_questions, 1):
            print(f"[{i}/{len(test_questions)}] {test_q['question']}")
            
            # Evaluate at k=3
            eval_3 = self.calculate_precision(
                test_q['question'], 
                test_q['expected_sources'],
                top_k=3
            )
            
            # Evaluate at k=5
            eval_5 = self.calculate_precision(
                test_q['question'],
                test_q['expected_sources'],
                top_k=5
            )
            
            # Calculate MRR
            mrr = self.calculate_mrr(
                test_q['question'],
                test_q['expected_sources'],
                top_k=10
            )
            
            # Get score statistics
            score_stats = self.get_relevance_scores(test_q['question'], top_k=5)
            
            # Store results
            results['precision_at_3'].append(eval_3['precision'])
            results['precision_at_5'].append(eval_5['precision'])
            results['recall_at_3'].append(eval_3['recall'])
            results['recall_at_5'].append(eval_5['recall'])
            results['f1_at_3'].append(eval_3['f1_score'])
            results['f1_at_5'].append(eval_5['f1_score'])
            results['mrr'].append(mrr)
            results['mean_scores'].append(score_stats['mean_score'])
            
            results['per_question_results'].append({
                'question': test_q['question'],
                'category': test_q['category'],
                'difficulty': test_q['difficulty'],
                'precision_at_3': eval_3['precision'],
                'precision_at_5': eval_5['precision'],
                'mrr': mrr,
                'retrieved_sources': eval_5['retrieved_sources']
            })
            
            print(f"  P@3: {eval_3['precision']:.2f} | P@5: {eval_5['precision']:.2f} | MRR: {mrr:.2f}\n")
        
        # Calculate aggregate metrics
        summary = {
            'mean_precision_at_3': np.mean(results['precision_at_3']),
            'mean_precision_at_5': np.mean(results['precision_at_5']),
            'mean_recall_at_3': np.mean(results['recall_at_3']),
            'mean_recall_at_5': np.mean(results['recall_at_5']),
            'mean_f1_at_3': np.mean(results['f1_at_3']),
            'mean_f1_at_5': np.mean(results['f1_at_5']),
            'mean_mrr': np.mean(results['mrr']),
            'mean_relevance_score': np.mean(results['mean_scores'])
        }
        
        return {
            'summary': summary,
            'detailed_results': results
        }


def main():
    """Run evaluation on the RAG system"""
    from src.retrieval.retriever import RAGRetriever
    
    print("="*60)
    print("RAG SYSTEM EVALUATION")
    print("="*60 + "\n")
    
    print("Initializing retriever...")
    retriever = RAGRetriever(top_k=5)
    
    evaluator = RAGEvaluator(retriever)
    
    results = evaluator.run_full_evaluation()
    
    # Display summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60 + "\n")
    
    summary = results['summary']
    
    print(f"Precision@3:        {summary['mean_precision_at_3']:.3f}")
    print(f"Precision@5:        {summary['mean_precision_at_5']:.3f}")
    print(f"Recall@3:           {summary['mean_recall_at_3']:.3f}")
    print(f"Recall@5:           {summary['mean_recall_at_5']:.3f}")
    print(f"F1@3:               {summary['mean_f1_at_3']:.3f}")
    print(f"F1@5:               {summary['mean_f1_at_5']:.3f}")
    print(f"Mean MRR:           {summary['mean_mrr']:.3f}")
    print(f"Mean Relevance:     {summary['mean_relevance_score']:.3f}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60 + "\n")
    
    print("These metrics measure:")
    print("- Retrieval accuracy at different k values")
    print("- How quickly the system finds relevant documents (MRR)")
    print("- Overall relevance of retrieved content")
    print("\nUse these numbers in your resume and documentation\n")


if __name__ == "__main__":
    main()