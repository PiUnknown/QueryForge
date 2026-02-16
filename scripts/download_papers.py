"""Automated paper download from ArXiv"""
import urllib.request
import os
from pathlib import Path

# Curated list of important RAG/LLM papers
PAPER_IDS = [
    # RAG Papers
    "2005.11401",  # Original RAG paper
    "2310.06825",  # RAG for LLMs Survey
    "2312.10997",  # Self-RAG
    "2401.15884",  # Corrective RAG
    
    # LLM Papers
    "2005.14165",  # GPT-3
    "2303.08774",  # GPT-4
    "2307.09288",  # Llama 2
    "2203.02155",  # InstructGPT
    
    # Embeddings & Retrieval
    "1810.04805",  # BERT
    "2104.08663",  # DPR (Dense Passage Retrieval)
    "2212.09741",  # Text Embeddings by Contrastive Pre-Training
    
    # Agents
    "2308.11432",  # ReAct: Reasoning and Acting
    "2303.17580",  # Reflexion
    "2308.10848",  # AutoGen
    
    # Prompting
    "2201.11903",  # Chain of Thought Prompting
    "2205.11916",  # Self-Consistency
    "2210.03493",  # ReAct Prompting
    
    # Additional Papers
    "2106.09685",  # LoRA
    "2305.14314",  # QLoRA  
    "2307.03172",  # LongLLaMA
    "2305.10601",  # Tree of Thoughts
]

def download_paper(arxiv_id, save_dir="data/raw"):
    """Download a paper from ArXiv"""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    filename = f"{arxiv_id}.pdf"
    filepath = os.path.join(save_dir, filename)
    
    if os.path.exists(filepath):
        print(f"[SKIP] {filename} already exists")
        return True
    
    try:
        print(f"[DOWN] Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        
        # Verify file size
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"[DONE] {filename} ({size_mb:.2f} MB)")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error downloading {arxiv_id}: {e}")
        return False

def main():
    # Create output directory
    save_dir = Path("data/raw")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(f"DOWNLOADING {len(PAPER_IDS)} RESEARCH PAPERS FROM ARXIV")
    print("="*60 + "\n")
    
    success_count = 0
    failed = []
    
    for i, paper_id in enumerate(PAPER_IDS, 1):
        print(f"[{i}/{len(PAPER_IDS)}] ", end="")
        if download_paper(paper_id, save_dir):
            success_count += 1
        else:
            failed.append(paper_id)
        print()
    
    print("="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Successfully downloaded: {success_count}/{len(PAPER_IDS)} papers")
    
    if failed:
        print(f"Failed: {len(failed)} papers")
        print(f"  Failed IDs: {', '.join(failed)}")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in save_dir.glob("*.pdf"))
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"\nTotal dataset size: {total_size_mb:.2f} MB")
    print(f"Location: {save_dir.absolute()}")
    print("\nData collection complete\n")

if __n