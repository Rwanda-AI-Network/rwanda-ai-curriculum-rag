# API Folder - Application Programming Interface

This folder contains all the API endpoints that external applications (web apps, mobile apps) can use to interact with the Rwanda AI Curriculum RAG system.

## What This Folder Does

Think of this as the **"front door"** to your backend system. It defines:
- **What requests** external apps can make
- **What data** needs to be sent with each request  
- **What responses** the system will send back
- **How authentication** and permissions work
- **How errors** are handled and reported

## Folder Structure

```
api/
├── __init__.py            # Package initialization
└── v1/                    # Version 1 of the API
    ├── __init__.py        # V1 router setup and organization
    ├── auth.py           # Authentication and user management
    ├── curriculum.py     # Curriculum content management
    ├── quiz.py           # Quiz generation and management
    ├── search.py         # Content search and retrieval
    ├── chat.py           # Conversational AI and chat
    └── admin.py          # Administrative operations
```

## API Version 1 (v1/) Explained

### auth.py - User Management
**Purpose**: Handle everything related to users
- **Registration** - New users signing up
- **Login/Logout** - User authentication
- **Profile Management** - Updating user information
- **Password Reset** - Forgotten password recovery
- **Role Management** - Teachers, students, admins, etc.

**Example Endpoints**:
- `POST /api/v1/auth/register` - Create new user account
- `POST /api/v1/auth/login` - User login with email/password
- `GET /api/v1/auth/profile` - Get current user's information
- `PUT /api/v1/auth/profile` - Update user profile

### curriculum.py - Content Management
**Purpose**: Manage educational content and documents
- **Upload Documents** - Add new curriculum materials (PDFs, Word docs, etc.)
- **Organize Content** - Categorize by subject, grade, topic
- **Search Documents** - Find specific curriculum materials
- **Content Metadata** - Track document information and usage

**Example Endpoints**:
- `POST /api/v1/curriculum/upload` - Upload new curriculum document
- `GET /api/v1/curriculum/documents` - List all documents with filtering
- `GET /api/v1/curriculum/subjects` - Get available subjects
- `DELETE /api/v1/curriculum/documents/{id}` - Remove document

### quiz.py - Assessment Generation
**Purpose**: AI-powered quiz and assessment creation
- **Generate Quizzes** - Create questions from curriculum content
- **Quiz Management** - Save, edit, and organize quizzes
- **Answer Checking** - Score student responses automatically
- **Analytics** - Track quiz performance and learning insights

**Example Endpoints**:
- `POST /api/v1/quiz/generate` - Create quiz from curriculum topic
- `GET /api/v1/quiz/{id}` - Get specific quiz with questions
- `POST /api/v1/quiz/{id}/attempt` - Submit quiz answers for grading
- `GET /api/v1/quiz/{id}/analytics` - Get quiz performance data

### search.py - Content Discovery
**Purpose**: Intelligent search across all educational content
- **Semantic Search** - AI-powered content understanding
- **Full-text Search** - Traditional keyword matching
- **Similar Content** - Find related materials
- **Search Analytics** - Track what users are looking for

**Example Endpoints**:
- `POST /api/v1/search/` - Search all content with AI understanding
- `GET /api/v1/search/suggestions` - Autocomplete search suggestions
- `POST /api/v1/search/similar` - Find content similar to a document
- `GET /api/v1/search/trending` - Popular search terms

### chat.py - Conversational AI
**Purpose**: Interactive educational conversations
- **AI Conversations** - Chat with AI tutor about curriculum topics
- **Context Awareness** - AI remembers conversation history
- **Curriculum Integration** - AI uses actual curriculum content in responses
- **Session Management** - Save and continue conversations

**Example Endpoints**:
- `POST /api/v1/chat/` - Send message to AI and get response
- `GET /api/v1/chat/sessions` - List user's conversation history
- `WebSocket /api/v1/chat/ws/{id}` - Real-time chat connection
- `POST /api/v1/chat/feedback/{id}` - Rate AI response quality

### admin.py - System Management
**Purpose**: Administrative tools and monitoring (admins only)
- **System Health** - Monitor application performance
- **User Management** - Manage user accounts and permissions
- **Content Moderation** - Review and approve content
- **Analytics** - System usage and performance reports

**Example Endpoints**:
- `GET /api/v1/admin/health` - Check system status
- `GET /api/v1/admin/users` - User management dashboard
- `POST /api/v1/admin/users/action` - Perform admin actions on users
- `GET /api/v1/admin/metrics` - System performance metrics

## For Contributors

### Implementation Approach
Each API file is a **comprehensive skeleton** with:

✅ **Complete Endpoint Definitions** - All major endpoints mapped out
📋 **Request/Response Models** - Data structures clearly defined
🔧 **Implementation Guides** - Step-by-step instructions in comments
🛡️ **Security Considerations** - Authentication and permission handling
📚 **Documentation** - Clear descriptions for each endpoint

### Getting Started
1. **Choose any API file** to start with
2. **Read the implementation guide** at the top of each endpoint
3. **Look at the TODO comments** for specific steps
4. **Implement one endpoint at a time** - they work independently
5. **Use the provided models** as guides for data structures

### Key Features Included
- **JWT Authentication** - Secure token-based authentication
- **Role-based Permissions** - Different access levels for different users
- **Request Validation** - Automatic checking of incoming data
- **Error Handling** - Consistent error responses
- **API Documentation** - Automatic generation of API docs
- **Backward Compatibility** - Support for existing endpoints

### Testing Your API
- **Swagger UI** available at `/docs` when running the server
- **Interactive testing** - Try endpoints directly in the browser
- **Request/response examples** - See exactly what data to send
- **Authentication testing** - Test login and protected endpoints

## Real-World Usage Examples

**Student Using Mobile App**:
1. `POST /auth/login` - Student logs in
2. `POST /search/` - Searches for "photosynthesis" 
3. `POST /chat/` - Asks AI tutor to explain photosynthesis
4. `POST /quiz/generate` - Generates practice quiz on photosynthesis

**Teacher Using Web Dashboard**:
1. `POST /auth/login` - Teacher logs in
2. `POST /curriculum/upload` - Uploads new lesson plan
3. `GET /quiz/analytics` - Reviews student quiz performance
4. `POST /admin/users/action` - Helps student with account issue

This API design makes the powerful backend accessible to any frontend application!