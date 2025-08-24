# Data Loader Folder - Content Import and Management

This folder handles importing curriculum content from various sources and formats into the Rwanda AI Curriculum RAG system.

## What This Folder Does

This is the **"intake system"** for educational content. It handles:
- **File Processing** - Reading PDFs, Word docs, text files, and more
- **Database Operations** - Both SQL and NoSQL database interactions
- **API Integration** - Connecting to external educational content sources
- **Data Cleaning** - Preparing content for AI processing
- **Content Organization** - Structuring imported content properly

## Folder Structure

```
data_loader/
├── __init__.py           # Package initialization
├── file_loader.py       # File processing (PDFs, Word docs, text files)
├── api_loader.py        # External API content loading
├── relational_db.py     # SQL database operations (PostgreSQL, MySQL, etc.)
├── nosql_db.py          # NoSQL database operations (MongoDB, etc.)
└── utils.py             # Shared utilities for data processing
```

## Files Explained

### file_loader.py - File Processing System
**Purpose**: Handle various document formats that contain curriculum content
- **PDF Processing** - Extract text, images, and structure from PDF files
- **Word Document Processing** - Handle DOCX files with formatting preservation
- **Text File Processing** - Process plain text, Markdown, and structured text
- **Image Processing** - Extract text from images using OCR (Optical Character Recognition)
- **Batch Processing** - Handle multiple files efficiently

**Supported File Types**:
- PDFs (curriculum textbooks, lesson plans)
- Word documents (DOCX format)
- Plain text files (TXT, MD)
- Rich text format (RTF)
- Images with text (JPG, PNG) using OCR

**Key Features**:
- **Metadata Extraction** - Author, creation date, subject information
- **Structure Preservation** - Keep headings, paragraphs, lists intact
- **Error Handling** - Graceful handling of corrupted or unsupported files
- **Progress Tracking** - Monitor processing of large documents

### api_loader.py - External Content Integration
**Purpose**: Connect to external educational content sources
- **Educational APIs** - Connect to curriculum databases and repositories
- **Government Sources** - Rwanda Education Board and ministry resources
- **International Content** - UNESCO, Khan Academy, and other educational platforms
- **Real-time Updates** - Automatically sync with external sources
- **Authentication** - Handle API keys and authentication for external services

**External Source Examples**:
- Rwanda Education Board curriculum database
- OpenStax educational resources
- Khan Academy content API
- UNESCO educational materials
- African educational content repositories

### relational_db.py - SQL Database Operations
**Purpose**: Handle structured data storage and retrieval
- **Curriculum Storage** - Store organized curriculum content with relationships
- **User Data** - Student progress, teacher information, administrative data
- **Query Optimization** - Efficient data retrieval for large datasets
- **Data Relationships** - Link subjects, grades, topics, and learning objectives
- **Transaction Management** - Ensure data consistency during complex operations

**Database Schema Areas**:
- **Curriculum Content** - Documents, lessons, topics, subjects
- **User Management** - Students, teachers, administrators
- **Learning Analytics** - Progress tracking, quiz results, engagement metrics
- **Content Relationships** - Prerequisites, learning paths, difficulty levels

### nosql_db.py - Document Database Operations  
**Purpose**: Handle flexible, unstructured content storage
- **Document Storage** - Store full curriculum documents with metadata
- **Content Indexing** - Prepare content for AI processing and search
- **Flexible Schemas** - Handle varying content structures without rigid schemas
- **High Performance** - Fast retrieval for AI processing and user queries
- **Content Versioning** - Track changes and updates to curriculum materials

**Use Cases**:
- **Raw Content Storage** - Original documents before processing
- **AI Embeddings** - Vector representations of content for semantic search
- **Session Data** - Chat conversations and user interaction history
- **Cache Storage** - Frequently accessed content for performance

### utils.py - Data Processing Utilities
**Purpose**: Shared functions for data cleaning and processing
- **Text Cleaning** - Remove formatting artifacts, normalize text
- **Language Detection** - Identify English vs Kinyarwanda content
- **Content Classification** - Automatically categorize by subject and grade level
- **Duplicate Detection** - Find and handle duplicate content
- **Quality Assessment** - Evaluate content completeness and quality

**Utility Functions**:
- Text preprocessing for AI models
- Content similarity detection
- Metadata extraction and validation
- File format conversion
- Error handling and logging

## For Contributors

### Implementation Status
This data loading system is **comprehensively designed** with:

✅ **Complete Processing Pipeline** - From raw files to AI-ready content
📄 **Multiple Format Support** - Handle diverse educational content formats
🔄 **External Integration** - Connect to educational content APIs
🗄️ **Database Flexibility** - Both SQL and NoSQL storage options
🛠️ **Implementation Guides** - Detailed instructions in each file

### Getting Started
1. **Start with `file_loader.py`** - Basic file processing capabilities
2. **Add `utils.py` functions** - Text processing and cleaning utilities  
3. **Implement database operations** - Choose SQL or NoSQL based on your needs
4. **Add API integrations** - Connect to external educational resources
5. **Test with sample files** - Use provided test curriculum documents

### Real-World Usage Examples

**Teacher Uploading Lesson Plans**:
1. Teacher uploads PDF lesson plan via web interface
2. `file_loader.py` extracts text and metadata
3. `utils.py` cleans and processes the content
4. `relational_db.py` stores structured information
5. `nosql_db.py` stores full document for AI processing

**System Importing Government Curriculum**:
1. `api_loader.py` connects to Rwanda Education Board API
2. Downloads official curriculum documents
3. `file_loader.py` processes downloaded files
4. Content is automatically categorized by subject and grade
5. Database stores organized curriculum for student access

**AI Processing Pipeline**:
1. Raw content loaded from various sources
2. `utils.py` cleans and normalizes text
3. Content is prepared for AI embedding generation
4. Processed content stored in optimized format
5. AI can quickly access content for question answering

### Key Features You'll Implement

**File Processing**:
- **Multi-format Support** - PDFs, Word docs, text files, images
- **Batch Processing** - Handle hundreds of documents efficiently
- **Error Recovery** - Continue processing even when some files fail
- **Progress Tracking** - Show upload/processing progress to users

**Database Operations**:
- **Flexible Storage** - Support both structured and unstructured data
- **Performance Optimization** - Fast queries for real-time AI responses
- **Data Consistency** - Reliable operations even with concurrent users
- **Backup and Recovery** - Protect valuable educational content

**External Integration**:
- **API Authentication** - Secure connections to external services
- **Rate Limiting** - Respect external service usage limits
- **Automatic Updates** - Keep curriculum content current
- **Error Handling** - Graceful handling of external service issues

### Testing Your Implementation
- **Sample Documents** - Test with various curriculum file formats
- **Database Testing** - Verify data integrity and query performance  
- **API Testing** - Mock external services for reliable testing
- **Performance Testing** - Handle large documents and concurrent users

This data loader system is the foundation that makes all AI features possible by ensuring high-quality, well-organized curriculum content!