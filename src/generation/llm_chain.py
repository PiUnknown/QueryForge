import requests
import json
from typing import List, Dict
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.retriever import RAGRetriever


class OllamaLLM:
    """Interface for interacting with Ollama's local LLM"""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        """
        Initialize connection to Ollama.
        
        Args:
            model: Name of the Ollama model to use
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Test connection
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"Connected to Ollama")
                print(f"  Model: {model}")
                print(f"  URL: {base_url}\n")
            else:
                print(f"Ollama connection issue: Status {response.status_code}")
        except Exception as e:
            print(f"Could not connect to Ollama: {e}")
            print("  Make sure Ollama is running")
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated text
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error generating response: {e}"


class RAGChain:
    """Complete RAG pipeline combining retrieval and generation"""
    
    def __init__(self, llm_model: str = "llama2", top_k: int = 5):
        """
        Initialize the RAG chain.
        
        Args:
            llm_model: Ollama model name (default: llama2)
            top_k: Number of chunks to retrieve
        """
        print("="*60)
        print("INITIALIZING RAG CHAIN")
        print("="*60 + "\n")
        
        print("Setting up retriever...")
        self.retriever = RAGRetriever(top_k=top_k)
        
        print("Setting up LLM...")
        self.llm = OllamaLLM(model=llm_model)
        
        print("="*60)
        print("RAG CHAIN READY")
        print("="*60 + "\n")
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt with context and query for the LLM.
        
        Args:
            query: User question
            context: Retrieved context from documents
        
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful AI assistant that answers questions based on provided research paper excerpts.

Context from research papers:
{context}

Question: {query}

Instructions:
- Answer based ONLY on the information in the context above
- If the context doesn't contain enough information, say so clearly
- Be specific and cite which sources support your answer
- Keep your answer concise but complete

Answer:"""
        
        return prompt
    
    def answer_question(self, query: str, top_k: int = None) -> Dict:
        """
        Execute the complete RAG pipeline to answer a question.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve (uses default if None)
        
        Returns:
            Dict containing query, answer, sources, and retrieved chunks
        """
        print(f"Query: {query}\n")
        
        # Step 1: Retrieve relevant chunks
        print("Retrieving relevant context...")
        chunks = self.retriever.retrieve(query, top_k=top_k)
        print(f"Retrieved {len(chunks)} chunks\n")
        
        # Step 2: Format context
        context = self.retriever.format_context(chunks)
        
        # Step 3: Create prompt
        prompt = self.create_prompt(query, context)
        
        # Step 4: Generate answer
        print("Generating answer...")
        answer = self.llm.generate(prompt, max_tokens=512)
        print("Answer generated\n")
        
        # Extract unique sources
        sources = list(set([
            chunk['metadata'].get('filename', 'Unknown') 
            for chunk in chunks
        ]))
        
        return {
            'query': query,
            'answer': answer,
            'sources': sources,
            'retrieved_chunks': chunks,
            'num_sources': len(chunks)
        }
    
    def interactive_session(self):
        """Run an interactive Q&A session"""
        print("="*60)
        print("INTERACTIVE RAG SESSION")
        print("="*60)
        print("Ask questions about the research papers")
        print("Type 'quit' or 'exit' to stop\n")
        
        while True:
            query = input("Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nSession ended")
                break
            
            if not query:
                continue
            
            print("\n" + "-"*60)
            
            result = self.answer_question(query, top_k=3)
            
            print("="*60)
            print("ANSWER:")
            print("="*60)
            print(result['answer'])
            
            print("\n" + "-"*60)
            print(f"Sources: {', '.join(result['sources'])}")
            print("-"*60 + "\n")


def main():
    """Test the RAG chain"""
    rag_chain = RAGChain(llm_model="llama2", top_k=5)
    
    test_queries = [
        "What is retrieval augmented generation?",
        "How do transformers process sequential data?",
        "What are the main components of BERT?"
    ]
    
    print("="*60)
    print("TESTING RAG CHAIN")
    print("="*60 + "\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST QUERY {i}/{len(test_queries)}")
        print(f"{'='*60}\n")
        
        result = rag_chain.answer_question(query, top_k=3)
        
        print("="*60)
        print("ANSWER:")
        print("="*60)
        print(result['answer'])
        
        print("\n" + "-"*60)
        print(f"Sources: {', '.join(result['sources'])}")
        print(f"Chunks used: {result['num_sources']}")
        print("-"*60 + "\n")
        
        if i < len(test_queries):
            input("Press Enter for next question...")
    
    print("\nTest complete\n")
    
    response = input("Start interactive Q&A session? (y/n): ")
    if response.lower() == 'y':
        print()
        rag_chain.interactive_session()


if __name__ == "__main__":
    main()