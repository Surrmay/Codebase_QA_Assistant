from typing import List
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from rich.console import Console
from rich.markdown import Markdown

console = Console()


class CodebaseQA:
    """Main QA assistant with conversational memory"""
    
    def __init__(self, vector_store_manager, groq_api_key: str):
        self.vector_store_manager = vector_store_manager
        
        # Initialize Groq LLM (free tier available)
        console.print("[blue]Initializing Groq LLM...[/blue]")
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",  # Free tier model
            temperature=0.2,
            max_tokens=2048
        )
        
        # Conversational memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Custom prompt for code Q&A
        # The chain expects a single input key when asking questions ("question").
        # We'll inject repository metadata into the question at runtime, so the
        # prompt template only needs: context (combined docs), chat_history, and question.
        self.prompt_template = """You are an expert code assistant analyzing a GitHub repository. 
Use the following code snippets to answer the user's question. Always cite the specific file paths when referencing code.

Code Context:
{context}

Chat History:
{chat_history}

Question: {question}

Instructions:
- Provide accurate, detailed answers based on the code provided
- Always mention specific file paths and line references when possible
- If you're not certain, say so
- Use code snippets to illustrate your points
- Be concise but thorough

Answer:"""

        self.PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        self.qa_chain = None
        console.print("[green]✓ QA Assistant initialized[/green]")
    
    def setup_chain(self):
        """Setup the conversational retrieval chain"""
        if not self.vector_store_manager.vector_store:
            raise ValueError("No vector store loaded. Please load or create one first.")
        
        # Create retriever from vector store
        retriever = self.vector_store_manager.vector_store.as_retriever(
            search_kwargs={"k": 6}  # Retrieve top 6 most relevant chunks
        )
        
        # Create conversational chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": self.PROMPT}
        )
        
        console.print("[green]✓ QA Chain configured[/green]")
    
    def ask(self, question: str) -> dict:
        """Ask a question about the codebase"""
        if not self.qa_chain:
            self.setup_chain()
        
        # Get repo metadata for context
        repo_info = self._format_repo_info()

        # The ConversationalRetrievalChain expects a single input key ("question").
        # Prepend repository metadata into the question text so the prompt still
        # receives repository information but the chain call uses one key.
        question_with_repo = f"Repository Information:\n{repo_info}\n\n{question}"

        # Query the chain using a single input key
        result = self.qa_chain({"question": question_with_repo})
        
        return {
            "answer": result["answer"],
            "source_documents": result["source_documents"],
            "chat_history": result.get("chat_history", [])
        }
    
    def _format_repo_info(self) -> str:
        """Format repository metadata for context"""
        metadata = self.vector_store_manager.repo_metadata
        repo_info = metadata.get('repo_info', {})
        
        info_str = f"Repository: {metadata.get('repo_name', 'Unknown')}\n"
        if repo_info:
            info_str += f"Description: {repo_info.get('description', 'N/A')}\n"
            info_str += f"Language: {repo_info.get('language', 'N/A')}\n"
            info_str += f"Total Files: {metadata.get('total_documents', 0)}\n"
        
        return info_str
    
    def display_answer(self, result: dict):
        """Display answer with rich formatting"""
        console.print("\n[bold cyan]Answer:[/bold cyan]")
        console.print(Markdown(result["answer"]))
        
        console.print("\n[bold yellow]📄 Source Files:[/bold yellow]")
        seen_files = set()
        for doc in result["source_documents"]:
            file_path = doc.metadata.get("file_path", "Unknown")
            if file_path not in seen_files:
                console.print(f"  • {file_path}")
                seen_files.add(file_path)
        
        console.print("\n" + "─" * 80 + "\n")
    
    def clear_memory(self):
        """Clear conversation history"""
        self.memory.clear()
        console.print("[yellow]Conversation memory cleared[/yellow]")
    
    def get_conversation_history(self) -> List:
        """Get current conversation history"""
        return self.memory.chat_memory.messages