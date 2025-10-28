🤖 Codebase Q&A Assistant with Contextual Memory
Ask natural language questions about any GitHub repository and get exact file/code references with AI-powered semantic search!

✨ Features
📦 Clone & Index any public GitHub repository
🔍 Semantic Search using RAG (Retrieval Augmented Generation)
🧠 Conversational Memory - maintains context across questions
📄 Source Citations - get exact file references
💯 100% Free - uses free tiers of Groq and HuggingFace
⚡ Fast Retrieval - FAISS vector store for instant results
🏗️ Architecture
User Question → Embeddings → FAISS Search → Relevant Code Chunks → Groq LLM → Answer + Citations
                    ↑                                                    ↓
                    └─────────── Conversation Memory ──────────────────┘
Components:

LangChain: Orchestration framework
Groq: Fast, free LLM inference (Mixtral-8x7B)
HuggingFace: Free embeddings (all-MiniLM-L6-v2)
FAISS: Local vector database for semantic search
GitHub API: Fetch repo metadata
🚀 Quick Start
1. Prerequisites
Python 3.8+
Git installed
Groq API key (free)
2. Installation
bash
# Clone this repository
git clone <your-repo-url>
cd codebase-qa-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
3. Configuration
Create a .env file:

bash
cp .env.example .env
Edit .env and add your API keys:

env
# Required: Get free API key from https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# Optional: For private repos and higher rate limits
# Get from https://github.com/settings/tokens
GITHUB_TOKEN=your_github_token_here
Getting Groq API Key (FREE):

Visit https://console.groq.com
Sign up (it's free!)
Go to API Keys section
Create new key
Copy to .env file
4. Run the Application
bash
python main.py
📖 Usage Guide
Index a Repository
Select option 1 from menu
Enter GitHub repository URL (e.g., https://github.com/username/repo)
Wait for indexing (1-5 minutes depending on repo size)
Repository is now ready for questions!
Ask Questions
Select option 2 from menu
Choose repository from list
Start asking questions!
Example Questions:

"What does the main function do?"
"How is authentication implemented?"
"Find all functions that handle database connections"
"Explain the API routing structure"
"Where is error handling implemented?"
"What libraries are used for data processing?"
Special Commands
Type quit - Exit chat session
Type clear - Clear conversation history
🎯 How It Works
1. Repository Parsing
python
# Clones repo and extracts code files
documents = repo_parser.parse_repo(repo_path)
# Filters: .py, .js, .ts, .java, .cpp, .md, etc.
# Skips: node_modules, .git, __pycache__, etc.
2. Document Chunking
python
# Splits code into semantic chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
3. Embedding & Indexing
python
# Converts chunks to vectors
embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)
4. Semantic Search + LLM
python
# Retrieves relevant chunks
relevant_docs = vector_store.similarity_search(query)

# Generates answer with context
answer = llm.generate(question + context)
📁 Project Structure
codebase-qa-assistant/
├── main.py                 # Entry point & CLI interface
├── repo_parser.py          # GitHub repo cloning & parsing
├── vector_store.py         # Embeddings & FAISS management
├── qa_assistant.py         # QA chain with memory
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
├── .env                   # Your API keys (gitignored)
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── repos/                # Cloned repositories (gitignored)
└── vector_stores/        # FAISS indices (gitignored)
🔧 Configuration Options
Supported File Types
Edit repo_parser.py to add more file extensions:

python
self.valid_extensions = {
    '.py', '.js', '.jsx', '.ts', '.tsx', 
    '.java', '.cpp', '.go', '.rs', ...
}
Chunk Size
Adjust in vector_store.py:

python
chunk_size=1000,      # Increase for more context
chunk_overlap=200,    # Increase for better continuity
Retrieval Count
Modify in qa_assistant.py:

python
search_kwargs={"k": 6}  # Number of chunks to retrieve
LLM Model
Change Groq model in qa_assistant.py:

python
model_name="mixtral-8x7b-32768"    # Default (fast & smart)
# model_name="llama2-70b-4096"     # Alternative
💡 Tips for Best Results
Ask Specific Questions: Instead of "How does this work?", ask "How is user authentication implemented?"
Reference File Names: "In the API module, how are requests handled?"
Follow-up Questions: The memory allows context-aware follow-ups: "What about error handling in that file?"
Code Exploration: "Find all classes that inherit from BaseModel"
Architecture Questions: "Explain the overall structure of the project"
🆓 Cost Breakdown
Component	Free Tier	This Project
Groq API	14,400 req/day	✅ Used
HuggingFace Models	Unlimited	✅ Used
FAISS	Local/Free	✅ Used
GitHub API	60 req/hr	✅ Used
Total Cost		$0/month
🚧 Limitations
Public repositories only (unless GitHub token provided)
Best with repos < 1000 files
English code comments work best
First indexing takes 1-5 minutes
🛠️ Troubleshooting
Error: "GROQ_API_KEY not found"

Make sure .env file exists with valid API key
Error: "No module named 'sentence_transformers'"

bash
pip install sentence-transformers
Slow indexing:

Normal for large repos
Progress bar shows status
One-time cost per repo
Out of memory:

Reduce chunk_size in vector_store.py
Reduce k value in retrieval
🔮 Future Enhancements
 Web interface (Streamlit/Gradio)
 Auto-sync with GitHub updates
 Multi-repo queries
 Export conversations
 Code diff analysis
 Custom model fine-tuning
📚 Learn More
LangChain Docs
Groq Documentation
FAISS Guide
RAG Explained
🤝 Contributing
Feel free to open issues or submit PRs!

📄 License
MIT License - feel free to use for any purpose!

Built with ❤️ using LangChain, Groq, and HuggingFace

