import pickle
from pathlib import Path
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from rich.console import Console

console = Console()

class VectorStoreManager:
    """Manages embeddings and vector store for code retrieval"""
    
    def __init__(self, persist_dir: str = "./vector_stores"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        
        # Use free HuggingFace embeddings (no API key needed)
        console.print("[blue]Loading embedding model...[/blue]")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        console.print("[green]✓ Embedding model loaded[/green]")
        
        # Text splitter for chunking code
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.vector_store = None
        self.repo_metadata = {}
    
    def create_vector_store(self, documents: List[Dict[str, str]], repo_name: str, repo_info: Dict = None):
        """Create vector store from parsed documents"""
        console.print(f"[blue]Creating vector store for {repo_name}...[/blue]")
        
        # Convert to LangChain documents with metadata
        langchain_docs = []
        for doc in documents:
            # Split content into chunks
            chunks = self.text_splitter.split_text(doc['content'])
            
            for i, chunk in enumerate(chunks):
                langchain_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        'file_path': doc['file_path'],
                        'file_name': doc['file_name'],
                        'extension': doc['extension'],
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                ))
        
        console.print(f"[blue]Creating embeddings for {len(langchain_docs)} chunks...[/blue]")
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(langchain_docs, self.embeddings)
        self.repo_metadata = {
            'repo_name': repo_name,
            'total_documents': len(documents),
            'total_chunks': len(langchain_docs),
            'repo_info': repo_info or {}
        }
        
        console.print(f"[green]✓ Vector store created with {len(langchain_docs)} chunks[/green]")
    
    def save_vector_store(self, repo_name: str):
        """Save vector store to disk"""
        if not self.vector_store:
            raise ValueError("No vector store to save")
        
        repo_dir = self.persist_dir / repo_name
        repo_dir.mkdir(exist_ok=True)
        
        # Save FAISS index
        self.vector_store.save_local(str(repo_dir / "faiss_index"))
        
        # Save metadata
        with open(repo_dir / "metadata.pkl", 'wb') as f:
            pickle.dump(self.repo_metadata, f)
        
        console.print(f"[green]✓ Vector store saved to {repo_dir}[/green]")
    
    def load_vector_store(self, repo_name: str):
        """Load vector store from disk"""
        repo_dir = self.persist_dir / repo_name
        
        if not repo_dir.exists():
            raise ValueError(f"Vector store for {repo_name} not found")
        
        console.print(f"[blue]Loading vector store for {repo_name}...[/blue]")
        
        # Load FAISS index
        try:
            # Try with the parameter (newer versions)
            self.vector_store = FAISS.load_local(
                str(repo_dir / "faiss_index"),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except TypeError:
            # Fall back without the parameter (older versions)
            self.vector_store = FAISS.load_local(
                str(repo_dir / "faiss_index"),
                self.embeddings
            )
        
        # Load metadata
        with open(repo_dir / "metadata.pkl", 'rb') as f:
            self.repo_metadata = pickle.load(f)
        
        console.print(f"[green]✓ Vector store loaded ({self.repo_metadata['total_chunks']} chunks)[/green]")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar code chunks"""
        if not self.vector_store:
            raise ValueError("No vector store loaded")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def list_available_stores(self) -> List[str]:
        """List all available vector stores"""
        return [d.name for d in self.persist_dir.iterdir() if d.is_dir()]