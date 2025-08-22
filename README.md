# Rwanda AI Curriculum RAG System

## Project Overview
The Rwanda AI Curriculum RAG System is a **Retrieval-Augmented Generation (RAG) system** for educational content. It helps learners access knowledge via AI-powered search and chat. The system can:

- Process documents from multiple sources
- Provide semantic search for relevant content
- Answer questions referencing original sources
- Maintain context in conversations for a smooth learning experience

This makes it easier for learners to explore educational materials intelligently and interactively.

---

## Table of Contents
1. [Getting Started](#getting-started)  
2. [Project Structure](#project-structure)  
3. [Installation](#installation)  
4. [Environment Configuration](#environment-configuration)  
5. [Running the Application](#running-the-application)  
6. [Contributing](#contributing)  
7. [License](#license)  

---

## Getting Started

### Prerequisites
- **Python 3.13+**  
- **Git**  
- **uv** package manager  

### Quick Start
1. Clone the repository:

   ```bash
   git clone https://github.com/RwandaAI/ai-service.git
   cd ai-service
   ```

2. Set up a virtual environment using uv:

   ```bash
   uv env create
   uv activate
   ```


3. Sync project dependencies:

   ```bash
   uv pip sync
   ```


4. Set up environment variables:

   ```bash
   cp .env.example app/config/env_files/.env
   ```

**Important**: Add your actual secret values in .env inside env_files/.
---

## Project Structure

```bash

rwanda-ai-curriculum-rag/
├── app/
│   ├── data_loader/              # Data ingestion layer
│   │   ├── __init__.py
│   │   ├── relational_db.py      # SQL DBs
│   │   ├── nosql_db.py           # MongoDB, Firebase
│   │   ├── file_loader.py        # CSV, TXT, PDF, DOCX
│   │   ├── api_loader.py         # External APIs
│   │   └── utils.py              # Shared helpers
│   │
│   ├── embeddings/               # Vectorization + vector DB
│   │   ├── vector_store.py
│   │   ├── create_embeddings.py
│   │   └── utils.py
│   │
│   ├── models/                   # LLMs + fine-tuning
│   │   ├── llm_inference.py
│   │   ├── fine_tune.py
│   │   ├── pipelines.py
│   │   └── utils.py
│   │
│   ├── prompts/                  # Prompt engineering
│   │   ├── learning_prompts.py
│   │   ├── quiz_prompts.py
│   │   └── utils.py
│   │
│   ├── services/                 # Core AI orchestration
│   │   ├── rag.py
│   │   ├── memory.py
│   │   └── response.py
│   │
│   ├── config/                   # Config & environment management
│   │   ├── __init__.py
│   │   ├── settings.py           # Reads from .env files
│   │   ├── constants.py          # Non-secret constants
│   │   ├── secrets.py            # Secure access helpers
│   │   └── env_files/           # Directory for all environment files
│   │       ├── .env.development
│   │       ├── .env.staging
│   │       ├── .env.production
│   │       └── .env.testing
│   │
│   ├── main.py                   # API entrypoint
│   └── logger.py                 # Centralized logging
│
├── tests/                        # Unit & integration tests
│   ├── test_data_loader.py
│   ├── test_embeddings.py
│   ├── test_models.py
│   ├── test_rag.py
│   └── test_response.py
│
├── utils/                        # Global utility helpers
│   ├── text_utils.py             # Text cleaning, tokenization
│   ├── evaluation.py             # Metrics (BLEU, ROUGE)
│   └── helper_functions.py       # Misc reusable functions
│
├── .env.example                  # Template (no secrets, safe for repo)
├── .gitignore                    # Ensure `.env` is ignored
├── requirements.txt
├── Dockerfile
├── docker-compose.yml            # For local orchestration (optional)
└── README.md

```

## Installation

1. Create and activate virtual environment using uv:

```bash
   uv venv
   
   # for linux
   . ./.venv/bin/activate
   # Or
   source ./.venv/bin/activate

   #for Windows
   ./.venv/scripts/activate
```

2. Sync dependencies:

```bash
uv sync
```


3. Set up environment variables:

```bash
cp .env.example app/config/env_files/.env
```


   - Add **real secret values** (API keys, tokens, DB passwords) in .env.

## Environment Configuration

- env_files/ stores environment files for different stages: development, staging, production, and testing.

- Config files roles:

  - settings.py: Reads the correct .env file for the current environment

  - constants.py: Stores non-secret constants used across the project

  - secrets.py: Provides safe access to sensitive keys and tokens

Keeping secrets in **env_files/** ensures they are secure and not committed to version control.

## Running the Application

### Development Mode

```bash
export DEBUG=True # Or in app/config/env_files/.env File
uv run app/main.py --reload
```

   - Hot-reloading enabled for rapid development

### Production Mode

```bash
export DEBUG=False # app/config/env_files/.env File
uv run app/main.py --host 0.0.0.0 --port 8000
```

   - Use uv or uvicorn for production-ready deployment

## Contributing

### We welcome contributions!

1. Fork the repository

2. Create a new branch:

```bash
git checkout -b feature/your-feature-name
```

3. Make your changes and run tests

4. Commit changes and push to your fork

5. Open a Pull Request

### Contribution types:

   1. Code
     - Bug fixes
     - New features
     - Performance improvements
     - Documentation updates


   2. Documentation
     - Improve README
     - Add code comments
     - Write tutorials

   3. Testing
     - Write unit tests
     - Add integration tests
     - Manual testing

### Code Style

  - Follow PEP 8 guidelines
  - Use type hints
  - Write docstrings for functions and classes
  - Keep functions focused and modular

### Git Workflow
  1. Create a branch from main
  2. Make your changes
  3. Write/update tests
  4. Update documentation
  5. Submit a Pull Request

### Testing

Run the test suite:

```bash
cd backend
pytest
``` 

Write tests for new features in the appropriate test file under app/tests/.



## License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

---

### © 2025 Rwanda AI Network

