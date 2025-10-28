import os
import shutil
from pathlib import Path
from typing import List, Dict
from git import Repo
from github import Github
from rich.console import Console
from rich.progress import track

console = Console()
class RepoParser:
    """Handles cloning and parsing GitHub repositories"""
    
    def __init__(self, github_token: str = None):
        self.github_token = github_token
        self.github_client = Github(github_token) if github_token else Github()

        self.valid_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h',
            '.cs', '.rb', '.go', '.rs', '.php', '.swift', '.kt', '.scala',
            '.md', '.txt', '.json', '.yaml', '.yml', '.toml', '.xml'
        }

        self.skip_dirs = {
            'node_modules', '.git', '__pycache__', 'venv', 'env',
            '.venv', 'dist', 'build', '.next', 'target', 'vendor'
        }
    
    def clone_repo(self, repo_url: str, local_path: str = "./repos") -> str:
        """Clone a GitHub repository locally"""
        try:
            repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
            clone_path = Path(local_path) / repo_name
            if clone_path.exists():
                console.print(f"[yellow]Removing existing repo at {clone_path}[/yellow]")
                shutil.rmtree(clone_path)
                
            console.print(f"[blue]Cloning repository: {repo_url}[/blue]")
            Repo.clone_from(repo_url, clone_path)
            console.print(f"[green]✓ Repository cloned to {clone_path}[/green]")
            
            return str(clone_path)
        
        except Exception as e:
            console.print(f"[red]Error cloning repository: {str(e)}[/red]")
            raise
    
    def parse_repo(self, repo_path: str) -> List[Dict[str, str]]:
        documents = []
        repo_path = Path(repo_path)
        console.print(f"[blue]Parsing repository: {repo_path.name}[/blue]")
        all_files = list(repo_path.rglob('*'))
        
        for file_path in track(all_files, description="Processing files..."):
            if not file_path.is_file():
                continue
                
            if any(skip_dir in file_path.parts for skip_dir in self.skip_dirs):
                continue
            
            if file_path.suffix not in self.valid_extensions:
                continue
            
            try:
                content = file_path.read_text(encoding='utf-8')
                relative_path = file_path.relative_to(repo_path)
                
                documents.append({
                    'content': content,
                    'file_path': str(relative_path),
                    'file_name': file_path.name,
                    'extension': file_path.suffix,
                    'size': len(content)
                })
                
            except Exception as e:
                console.print(f"[yellow]Skipping {file_path.name}: {str(e)}[/yellow]")
                continue
        
        console.print(f"[green]✓ Parsed {len(documents)} files[/green]")
        return documents
    
    def get_repo_info(self, repo_url: str) -> Dict:
        """Fetch repository metadata from GitHub API"""
        try:
            parts = repo_url.rstrip('/').replace('.git', '').split('/')
            owner, repo_name = parts[-2], parts[-1]
            
            repo = self.github_client.get_repo(f"{owner}/{repo_name}")
            
            return {
                'name': repo.name,
                'full_name': repo.full_name,
                'description': repo.description or "No description available",
                'language': repo.language,
                'stars': repo.stargazers_count,
                'forks': repo.forks_count,
                'url': repo.html_url,
                'last_updated': repo.updated_at.isoformat()
            }
        
        except Exception as e:
            console.print(f"[yellow]Could not fetch repo metadata: {str(e)}[/yellow]")
            return {'name': 'Unknown', 'description': 'No metadata available'}