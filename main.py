import os
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from repo_parser import RepoParser
from vector_store import VectorStoreManager
from qa_assistant import CodebaseQA

console = Console()


def display_banner():
    """Display welcome banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║        🤖 Codebase Q&A Assistant with Memory 🧠              ║
║                                                               ║
║        Ask questions about any GitHub repository!             ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold blue"))


def main():
    load_dotenv()
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not groq_api_key:
        console.print("[red]Error: GROQ_API_KEY not found in .env file[/red]")
        console.print("Get your free API key from: https://console.groq.com")
        return
    
    display_banner()
    repo_parser = RepoParser(github_token=github_token)
    vector_store_manager = VectorStoreManager()
    qa_assistant = CodebaseQA(vector_store_manager, groq_api_key)
    while True:
        console.print("\n[bold cyan]Options:[/bold cyan]")
        console.print("1. 📦 Index a new GitHub repository")
        console.print("2. 💬 Chat with an existing repository")
        console.print("3. 📋 List indexed repositories")
        console.print("4. 🚪 Exit")
        
        choice = Prompt.ask("\nSelect an option", choices=["1", "2", "3", "4"])
        
        if choice == "1":
            index_new_repository(repo_parser, vector_store_manager)
        
        elif choice == "2":
            chat_with_repository(vector_store_manager, qa_assistant)
        
        elif choice == "3":
            list_repositories(vector_store_manager)
        
        elif choice == "4":
            console.print("\n[green]Thanks for using Codebase Q&A! 👋[/green]")
            break


def index_new_repository(repo_parser: RepoParser, vector_store_manager: VectorStoreManager):
    console.print("\n[bold]Index New Repository[/bold]")
    
    repo_url = Prompt.ask("Enter GitHub repository URL")
    
    try:
        repo_path = repo_parser.clone_repo(repo_url)
        
        documents = repo_parser.parse_repo(repo_path)
        
        if not documents:
            console.print("[red]No valid documents found in repository[/red]")
            return
        repo_info = repo_parser.get_repo_info(repo_url)
        
        repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
        vector_store_manager.create_vector_store(documents, repo_name, repo_info)
        
        vector_store_manager.save_vector_store(repo_name)
        
        console.print(f"\n[green]✅ Repository '{repo_name}' successfully indexed![/green]")
        
    except Exception as e:
        console.print(f"[red]Error indexing repository: {str(e)}[/red]")


def chat_with_repository(vector_store_manager: VectorStoreManager, qa_assistant: CodebaseQA):
    available_repos = vector_store_manager.list_available_stores()
    
    if not available_repos:
        console.print("[yellow]No repositories indexed yet. Please index a repository first.[/yellow]")
        return
    
    console.print("\n[bold]Available Repositories:[/bold]")
    for i, repo in enumerate(available_repos, 1):
        console.print(f"{i}. {repo}")
    
    choice = Prompt.ask("Select repository number", choices=[str(i) for i in range(1, len(available_repos) + 1)])
    repo_name = available_repos[int(choice) - 1]
    
    try:
        vector_store_manager.load_vector_store(repo_name)
        qa_assistant.setup_chain()
        
        console.print(f"\n[green]✅ Loaded repository: {repo_name}[/green]")
        console.print("[dim]Type 'quit' to exit, 'clear' to clear conversation history[/dim]\n")

        while True:
            question = Prompt.ask("\n[bold cyan]Your question[/bold cyan]")
            
            if question.lower() == 'quit':
                break
            
            if question.lower() == 'clear':
                qa_assistant.clear_memory()
                continue
            
            if not question.strip():
                continue
            
            console.print("\n[dim]Thinking...[/dim]")
            result = qa_assistant.ask(question)
            qa_assistant.display_answer(result)
    
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")


def list_repositories(vector_store_manager: VectorStoreManager):
    """List all indexed repositories"""
    available_repos = vector_store_manager.list_available_stores()
    
    if not available_repos:
        console.print("\n[yellow]No repositories indexed yet.[/yellow]")
    else:
        console.print("\n[bold]Indexed Repositories:[/bold]")
        for i, repo in enumerate(available_repos, 1):
            console.print(f"{i}. {repo}")


if __name__ == "__main__":
    main()