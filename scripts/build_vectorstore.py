"""
Build FAISS vector store from Nietzsche's texts.
This script should be run once to create the vector store.
"""
import os
import glob
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vectorstore():
    """Load all Nietzsche texts, chunk them, and create FAISS vector store."""
    
    print("Loading Nietzsche texts...")
    
    # Get all text files from preprocessed directory
    text_dir = Path("content/nietzsche/preprocessed")
    text_files = glob.glob(str(text_dir / "*.txt"))
    
    if not text_files:
        raise FileNotFoundError(f"No text files found in {text_dir}")
    
    print(f"Found {len(text_files)} texts")
    
    # Load all documents
    documents = []
    for file_path in text_files:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            
            # Add metadata about the source work
            work_name = Path(file_path).stem.replace('_', ' ').title()
            for doc in docs:
                doc.metadata['source'] = work_name
                doc.metadata['file'] = os.path.basename(file_path)
            
            documents.extend(docs)
            print(f"  ✓ Loaded {work_name}")
        except Exception as e:
            print(f"  ✗ Error loading {file_path}: {e}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    
    # Split documents into chunks
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Create embeddings
    print("\nCreating embeddings (this may take a few minutes)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Build FAISS vector store
    print("\nBuilding FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save to disk
    vectorstore_dir = Path("vectorstore")
    vectorstore_dir.mkdir(exist_ok=True)
    vectorstore.save_local(str(vectorstore_dir))
    
    print(f"\n✓ Vector store successfully saved to {vectorstore_dir}/")
    print(f"  Total chunks indexed: {len(chunks)}")
    print(f"  Ready to use in the Streamlit app!")

if __name__ == "__main__":
    build_vectorstore()

