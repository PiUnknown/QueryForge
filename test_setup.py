"""Test script to verify all dependencies are installed correctly"""

def test_imports():
    print("="*60)
    print("TESTING IMPORTS")
    print("="*60 + "\n")
    
    tests = [
        ("langchain", "LangChain"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
        ("pypdf", "PyPDF"),
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("dotenv", "Python-dotenv"),
    ]
    
    success_count = 0
    for module, name in tests:
        try:
            __import__(module)
            print(f"[PASS] {name:30} imported successfully")
            success_count += 1
        except ImportError as e:
            print(f"[FAIL] {name:30} FAILED: {e}")
    
    print(f"\n{success_count}/{len(tests)} imports successful\n")

def test_ollama():
    print("="*60)
    print("TESTING OLLAMA (Free Local LLM)")
    print("="*60 + "\n")
    
    import subprocess
    
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        
        if result.returncode == 0:
            print("[PASS] Ollama is installed and running")
            print("\nInstalled models:")
            print(result.stdout)
            
            if 'llama2' in result.stdout:
                print("[PASS] Llama2 model ready to use")
            else:
                print("[WARN] Llama2 not found. Run: ollama pull llama2")
        else:
            print("[FAIL] Ollama installed but not responding")
            
    except FileNotFoundError:
        print("[FAIL] Ollama not installed")
        print("  Download from: https://ollama.ai/")
    except Exception as e:
        print(f"[FAIL] Error checking Ollama: {e}")
    
    print("\n")

def test_embeddings():
    print("="*60)
    print("TESTING FREE EMBEDDINGS")
    print("="*60 + "\n")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("[INFO] Loading free embedding model...")
        
        # This will download the model first time (~90MB)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test embedding
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        
        print(f"[PASS] Embeddings working (Dimension: {len(embedding)})")
        
    except Exception as e:
        print(f"[FAIL] Embedding test failed: {e}")
    
    print("\n")

def test_folder_structure():
    print("="*60)
    print("TESTING FOLDER STRUCTURE")
    print("="*60 + "\n")
    
    import os
    
    required_folders = [
        "data/raw",
        "data/processed",
        "src/ingestion",
        "src/retrieval",
        "src/generation",
        "src/evaluation",
        "src/utils",
        "experiments",
    ]
    
    for folder in required_folders:
        if os.path.exists(folder):
            print(f"[PASS] {folder}")
        else:
            print(f"[FAIL] {folder} - MISSING")
    
    print("\n")

def test_config():
    print("="*60)
    print("TESTING CONFIGURATION")
    print("="*60 + "\n")
    
    try:
        from src.utils.config import Config
        
        print(f"[INFO] USE_LOCAL: {Config.USE_LOCAL}")
        print(f"[INFO] Embedding Model: {Config.EMBEDDING_MODEL}")
        print(f"[INFO] LLM Model: {Config.LLM_MODEL}")
        print(f"[INFO] Chunk Size: {Config.CHUNK_SIZE}")
        print(f"[INFO] Top K: {Config.TOP_K}")
        
    except Exception as e:
        print(f"[FAIL] Config error: {e}")
    
    print("\n")

def main():
    print("\n")
    print("="*60)
    print("RESEARCH RAG SYSTEM - SETUP VERIFICATION")
    print("="*60)
    print("\n")
    
    test_folder_structure()
    test_config()
    test_imports()
    test_embeddings()
    test_ollama()
    
    print("="*60)
    print("SETUP VERIFICATION COMPLETE")
    print("="*60)
    print("\nTotal Cost: $0 - Everything runs locally")
    print("\nYou're ready to proceed to data collection\n")

if __name__ == "__main__":
    main()