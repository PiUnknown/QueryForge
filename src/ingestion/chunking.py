from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from pathlib import Path


class DocumentChunker:
    """Implements different text chunking strategies for RAG"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def fixed_size_chunking(self, text: str, metadata: Dict) -> List[Dict]:
        """Strategy 1: Fixed-size chunks with simple character splitting"""
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n\n"
        )
        
        chunks = splitter.split_text(text)
        
        return [
            {
                'text': chunk,
                'metadata': {
                    **metadata,
                    'chunk_id': i,
                    'strategy': 'fixed_size',
                    'chunk_size': len(chunk)
                }
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def recursive_chunking(self, text: str, metadata: Dict) -> List[Dict]:
        """Strategy 2: Recursive splitting by paragraphs, sentences, then characters"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_text(text)
        
        return [
            {
                'text': chunk,
                'metadata': {
                    **metadata,
                    'chunk_id': i,
                    'strategy': 'recursive',
                    'chunk_size': len(chunk)
                }
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def semantic_chunking(self, text: str, metadata: Dict) -> List[Dict]:
        """Strategy 3: Sentence-aware semantic chunking"""
        sentences = text.split('. ')
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '. '
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return [
            {
                'text': chunk,
                'metadata': {
                    **metadata,
                    'chunk_id': i,
                    'strategy': 'semantic',
                    'chunk_size': len(chunk)
                }
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def chunk_document(self, text: str, metadata: Dict, strategy: str = "all") -> Dict:
        """
        Chunk a document using specified strategy.
        
        Args:
            text: Document text
            metadata: Document metadata
            strategy: 'fixed', 'recursive', 'semantic', or 'all'
        
        Returns:
            Dict with chunks for each strategy
        """
        result = {
            'fixed_size': [],
            'recursive': [],
            'semantic': []
        }
        
        if strategy in ["fixed", "all"]:
            result['fixed_size'] = self.fixed_size_chunking(text, metadata)
        
        if strategy in ["recursive", "all"]:
            result['recursive'] = self.recursive_chunking(text, metadata)
        
        if strategy in ["semantic", "all"]:
            result['semantic'] = self.semantic_chunking(text, metadata)
        
        return result


def main():
    """Test chunking strategies"""
    print("="*60)
    print("TEXT CHUNKING TEST")
    print("="*60 + "\n")
    
    # Load sample document
    processed_dir = Path("data/processed")
    txt_files = list(processed_dir.glob("*.txt"))
    
    if not txt_files:
        print("No processed files found. Run pdf_loader.py first.")
        return
    
    sample_file = txt_files[0]
    print(f"Testing on: {sample_file.name}\n")
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    metadata = {'filename': sample_file.name, 'source': str(sample_file)}
    
    # Test all strategies
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=100)
    result = chunker.chunk_document(text, metadata, strategy="all")
    
    # Display results
    print("="*60)
    print("RESULTS")
    print("="*60 + "\n")
    
    for strategy_name, chunks in result.items():
        if chunks:
            print(f"Strategy: {strategy_name.upper()}")
            print(f"  Total chunks: {len(chunks)}")
            
            avg_size = sum(c['metadata']['chunk_size'] for c in chunks) / len(chunks)
            print(f"  Average size: {avg_size:.0f} characters")
            
            print(f"  Sample (first 200 chars):")
            print(f"    {chunks[0]['text'][:200]}...")
            print()
    
    # Comparison
    print("="*60)
    print("COMPARISON")
    print("="*60 + "\n")
    
    print(f"{'Strategy':<15} {'Chunks':<10} {'Avg Size':<12} {'Coverage'}")
    print("-"*60)
    
    for strategy_name, chunks in result.items():
        if chunks:
            total_chars = sum(c['metadata']['chunk_size'] for c in chunks)
            avg_size = total_chars / len(chunks)
            coverage = (total_chars / len(text)) * 100
            
            print(f"{strategy_name:<15} {len(chunks):<10} {avg_size:<12.0f} {coverage:.1f}%")
    
    print("\nTest complete\n")


if __name__ == "__main__":
    main()