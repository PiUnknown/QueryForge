import os
from pathlib import Path
from typing import List, Dict
import pypdf
from tqdm import tqdm


class PDFLoader:
    """Handles PDF text extraction with metadata preservation"""
    
    def __init__(self, pdf_directory: str = "data/raw"):
        self.pdf_dir = Path(pdf_directory)
        
    def extract_text(self, pdf_path: Path) -> Dict:
        """
        Extract text from a single PDF file.
        Returns dict with metadata and extracted text by page.
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                
                metadata = {
                    'filename': pdf_path.name,
                    'num_pages': len(reader.pages),
                    'source': str(pdf_path),
                }
                
                pages = []
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if text.strip():
                        pages.append({
                            'page_num': page_num,
                            'text': text
                        })
                
                full_text = '\n\n'.join([p['text'] for p in pages])
                
                return {
                    'metadata': metadata,
                    'pages': pages,
                    'full_text': full_text
                }
                
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            return None
    
    def load_all(self) -> List[Dict]:
        """Load and extract text from all PDFs in directory"""
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_dir}")
            return []
        
        print(f"Found {len(pdf_files)} PDF files")
        print("Extracting text...\n")
        
        documents = []
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            doc = self.extract_text(pdf_path)
            if doc:
                documents.append(doc)
        
        print(f"\nProcessed {len(documents)} documents successfully")
        return documents
    
    def save_text(self, documents: List[Dict], output_dir: str = "data/processed"):
        """Save extracted text to individual files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for doc in documents:
            filename = doc['metadata']['filename'].replace('.pdf', '.txt')
            output_file = output_path / filename
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(doc['full_text'])
        
        print(f"Saved text files to {output_path}")


def main():
    """Test the PDF loader"""
    print("="*60)
    print("PDF TEXT EXTRACTION")
    print("="*60 + "\n")
    
    loader = PDFLoader("data/raw")
    documents = loader.load_all()
    
    if documents:
        loader.save_text(documents)
        
        # Show statistics
        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        
        total_pages = sum(doc['metadata']['num_pages'] for doc in documents)
        total_chars = sum(len(doc['full_text']) for doc in documents)
        
        print(f"Documents: {len(documents)}")
        print(f"Pages: {total_pages}")
        print(f"Characters: {total_chars:,}")
        print(f"Average pages/doc: {total_pages / len(documents):.1f}")
        
        # Sample output
        print("\n" + "="*60)
        print("SAMPLE (first 500 characters)")
        print("="*60)
        print(documents[0]['full_text'][:500] + "...")
        
        print("\nExtraction complete\n")
    else:
        print("\nNo documents were processed")


if __name__ == "__main__":
    main()