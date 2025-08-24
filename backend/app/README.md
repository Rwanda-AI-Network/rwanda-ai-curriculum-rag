# App Folder - Main Application Code

This folder contains the core application code for the Rwanda AI Curriculum RAG system.

## What This Folder Does

This is where all the main application logic lives. Think of it as the "brain" of the system that handles:
- **API requests** from web or mobile apps
- **AI processing** for educational content
- **Database operations** for storing and retrieving data
- **User authentication** and permissions
- **Content management** for curriculum materials

## Folder Structure

```
app/
├── main.py                 # Application startup and main configuration
├── logger.py              # Logging system for tracking what happens
├── api/                   # API endpoints (what external apps can call)
├── config/                # Settings and configuration management  
├── data_loader/           # Loading curriculum content from files/databases
├── embeddings/            # AI vector processing for content search
├── models/                # AI models and machine learning processing
├── prompts/               # AI prompt templates for different tasks
├── services/              # Core business logic and service classes
```

## Key Components Explained

### main.py - Application Entry Point
- **Starts the entire system** when you run the backend
- **Configures all settings** and middleware
- **Sets up API routes** so external apps can communicate
- **Handles errors** and logging across the system
- **Manages application lifecycle** (startup and shutdown)

### api/ - External Interface
Contains all the endpoints that web/mobile apps can call:
- **auth/** - User login, registration, password management
- **curriculum/** - Upload, manage, and organize curriculum documents
- **quiz/** - Generate and manage AI-powered quizzes
- **search/** - Search through curriculum content intelligently
- **chat/** - Conversational AI for educational assistance
- **admin/** - System management and monitoring tools

### services/ - Core Business Logic  
The main processing engines:
- **rag.py** - Retrieval Augmented Generation (AI that finds and uses relevant content)
- **response.py** - Formats AI responses appropriately for users
- **memory.py** - Manages conversation history and context

### Other Important Folders
- **config/** - Manages settings for different environments (development, production)
- **data_loader/** - Handles importing curriculum from various file formats
- **embeddings/** - Converts text into AI-understandable vector representations
- **models/** - AI model management and inference processing
- **prompts/** - Template messages for different AI tasks

## For Contributors

### Implementation Status
This folder contains **skeleton code** with comprehensive implementation guides:

✅ **Fully Mapped Out** - Every major component is planned and documented
🔨 **Mock Classes Provided** - System runs during development
📚 **Implementation Guides** - Detailed comments explain what to build
🧪 **Tests Available** - Validation for your implementations

### Where to Start
1. **Pick any component** - they're designed to work independently
2. **Read the implementation guides** in each file
3. **Replace mock classes** with real functionality
4. **Use the tests** to validate your work
5. **Follow the TODO comments** for step-by-step guidance

### Key Features You'll Implement
- **AI-powered content processing** using vector embeddings
- **Intelligent quiz generation** from curriculum materials  
- **Conversational AI** with educational context awareness
- **Multi-language support** (English and Kinyarwanda)
- **Role-based access control** for different user types
- **Real-time chat** with WebSocket connections

## File Overview

| File/Folder | Purpose | Implementation Status |
|-------------|---------|----------------------|
| `main.py` | Application startup and configuration | ✅ Complete skeleton |
| `logger.py` | System-wide logging | ✅ Ready to implement |
| `api/` | REST API endpoints | ✅ Comprehensive structure |
| `config/` | Configuration management | ✅ Settings framework |
| `services/` | Core business logic | 🔨 Service interfaces ready |
| `data_loader/` | Content import system | 🔨 Loader framework |
| `embeddings/` | AI vector processing | 🔨 Vector store interface |
| `models/` | AI model management | 🔨 Model interfaces |
| `prompts/` | AI prompt templates | ✅ Template structure |

This folder is designed to be both comprehensive and approachable - dive in anywhere and start building!