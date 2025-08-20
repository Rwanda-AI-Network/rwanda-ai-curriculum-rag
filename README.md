# Rwanda AI Curriculum RAG System

A Retrieval-Augmented Generation (RAG) system for educational content, built by the Rwanda AI Network. This system helps make educational content more accessible through AI-powered search and chat interfaces.

## Project Overview

This project combines document processing, vector search, and AI to create an intelligent system that can:
- Process and index educational documents (PDFs, text files)
- Provide semantic search across educational content
- Answer questions using AI with references to source materials
- Maintain conversation context for better learning interactions

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the Application](#running-the-application)
6. [Contributing](#contributing)
7. [License](#license)

## Getting Started

### Prerequisites

- Python 3.13 or higher
- Git
- uv package manager

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/Rwanda-AI-Network/rwanda-ai-curriculum-rag.git
cd rwanda-ai-curriculum-rag
```

2. Set up the virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
cd backend
uv pip sync
```

4. Copy the example environment file:
```bash
cp .env.example .env
```

5. Update the `.env` file with your configuration

6. Run the application:
```bash
python main.py
```

## Project Structure

The project is organized into two main parts:

### Backend (`/backend`)
- `app/`: Main application code
  - `api/`: REST API endpoints
  - `core/`: Core functionality and configuration
  - `db/`: Database and storage connections
  - `ingestions/`: Document processing and indexing
  - `schemas/`: Data models and validation
  - `services/`: Business logic and AI services
  - `tests/`: Test suites

### Frontend (`/frontend`)
- User interface code (to be implemented)

## Installation

### Backend Setup

1. Install Python dependencies:
```bash
cd backend
uv pip sync
```

2. Configure environment variables:
- Copy `.env.example` to `.env`
- Update the following variables:
  - `DATABASE_URL`: Vector database connection string
  - `OPENAI_API_KEY`: OpenAI API key (if using OpenAI)
  - `MODEL_PATH`: Path to local models (if using HuggingFace)

### Frontend Setup

Frontend implementation is planned for future development.

## Configuration

The application uses environment variables for configuration. Key settings:

- `ENV`: Environment (development/production)
- `DEBUG`: Enable debug mode
- `LOG_LEVEL`: Logging level
- `DATABASE_URL`: Vector database connection URL
- `EMBEDDING_PROVIDER`: Choice of embedding service
- `LLM_PROVIDER`: Choice of language model service

See `.env.example` for all available options.

## Running the Application

### Development Mode

```bash
cd backend
python main.py
```

### Production Mode

For production deployment:

1. Set environment variables:
```bash
export ENV=production
export DEBUG=false
```

2. Run with production server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Contributing

We welcome contributions of all kinds! Here's how you can help:

### Setting Up for Development

1. Fork the repository
2. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

3. Make your changes
4. Run tests:
```bash
cd backend
pytest
```

5. Submit a Pull Request

### Types of Contributions

1. **Code**
   - Bug fixes
   - New features
   - Performance improvements
   - Documentation updates

2. **Documentation**
   - Improve README
   - Add code comments
   - Write tutorials

3. **Testing**
   - Write unit tests
   - Add integration tests
   - Manual testing

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for functions and classes
- Keep functions focused and modular

### Git Workflow

1. Create a branch from `main`
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

Write tests for new features in the appropriate test file under `app/tests/`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

© 2025 Rwanda AI Network