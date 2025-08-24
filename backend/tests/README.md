# Tests Folder - Quality Assurance and Validation

This folder contains automated tests that ensure the Rwanda AI Curriculum RAG system works correctly, reliably, and safely.

## What This Folder Does

This is the **"quality control department"** that:
- **Verifies Functionality** - Ensures all features work as designed
- **Prevents Regressions** - Catches when changes break existing features
- **Validates Educational Quality** - Confirms AI responses are pedagogically sound
- **Tests Performance** - Ensures system handles expected load
- **Ensures Safety** - Validates appropriate content for educational settings

Think of tests as the "quality inspectors" that check every part of the system before students and teachers use it.

## Folder Structure

```
tests/
├── test_data_loader.py    # Test file processing and data import functionality
├── test_embeddings.py     # Test AI vector generation and semantic search
├── test_models.py         # Test AI model integration and responses
├── test_rag.py           # Test core RAG (Retrieval Augmented Generation) logic
└── test_response.py      # Test response formatting and delivery
```

## Files Explained

### test_data_loader.py - Content Import Testing
**Purpose**: Verify that curriculum content is loaded, processed, and stored correctly

**What It Tests**:
- **File Processing** - PDFs, Word docs, text files are read correctly
- **Content Extraction** - Text, metadata, structure are properly extracted
- **Database Storage** - Content is saved with correct relationships and indexing
- **Error Handling** - System handles corrupted files and missing data gracefully
- **Batch Processing** - Multiple files processed efficiently without errors

**Test Examples**:
```python
def test_pdf_processing():
    """Test that PDF curriculum files are processed correctly."""
    # Load sample curriculum PDF
    # Verify text extraction
    # Check metadata extraction (subject, grade, topic)
    # Ensure proper content structure

def test_duplicate_content_handling():
    """Test system handles duplicate curriculum documents."""
    # Load same document twice
    # Verify duplicate detection
    # Check that only one copy is stored

def test_batch_upload_performance():
    """Test system can handle multiple file uploads efficiently."""
    # Upload 50+ curriculum documents
    # Measure processing time
    # Verify all documents processed correctly
```

### test_embeddings.py - AI Vector Processing Testing
**Purpose**: Verify that curriculum content is converted to AI-understandable vectors correctly

**What It Tests**:
- **Vector Generation** - Text content creates meaningful vector representations
- **Similarity Matching** - Related content has similar vectors
- **Search Accuracy** - Vector search finds relevant content for queries
- **Performance** - Vector operations complete within acceptable time
- **Multilingual Support** - English and Kinyarwanda content processed correctly

**Test Examples**:
```python
def test_semantic_similarity():
    """Test that similar curriculum topics have similar vectors."""
    # Create vectors for "photosynthesis" and "plant nutrition"
    # Verify high similarity score
    # Test that unrelated topics have low similarity

def test_search_relevance():
    """Test that vector search returns relevant results."""
    # Search for "water cycle"
    # Verify results include evaporation, condensation, precipitation content
    # Check that irrelevant topics (like mathematics) are not returned

def test_kinyarwanda_processing():
    """Test vector generation for Kinyarwanda curriculum content."""
    # Process Kinyarwanda text
    # Verify vectors are generated
    # Test cross-language search capabilities
```

### test_models.py - AI Model Testing
**Purpose**: Verify AI models generate appropriate educational responses

**What It Tests**:
- **Response Quality** - AI generates educationally sound content
- **Grade Appropriateness** - Responses match targeted grade level
- **Cultural Sensitivity** - AI uses appropriate Rwandan context
- **Safety Filtering** - Inappropriate content is filtered out
- **Performance** - AI responses generated within acceptable time

**Test Examples**:
```python
def test_grade_level_adaptation():
    """Test AI adjusts explanations for different grade levels."""
    # Ask same question for P2 and S4 students
    # Verify P2 response uses simpler language
    # Check S4 response includes more complex concepts

def test_cultural_context():
    """Test AI includes appropriate Rwandan examples."""
    # Ask about agriculture
    # Verify response mentions local crops (bananas, cassava, tea)
    # Check for culturally appropriate examples

def test_educational_safety():
    """Test AI responses are appropriate for educational settings."""
    # Test various potentially sensitive topics
    # Verify responses are age-appropriate
    # Check that harmful content is filtered out
```

### test_rag.py - Core Intelligence Testing
**Purpose**: Verify the core RAG system combines content retrieval with AI generation correctly

**What It Tests**:
- **Content Retrieval** - System finds relevant curriculum content for questions
- **Context Integration** - Multiple content sources combined coherently
- **Response Accuracy** - AI responses are based on actual curriculum content
- **Conversation Context** - System remembers and uses conversation history
- **Educational Alignment** - Responses support curriculum learning objectives

**Test Examples**:
```python
def test_curriculum_grounded_responses():
    """Test that AI responses are based on actual curriculum content."""
    # Ask question about specific curriculum topic
    # Verify response content matches curriculum documents
    # Check that AI doesn't hallucinate facts not in curriculum

def test_conversation_continuity():
    """Test system maintains context across conversation turns."""
    # Start conversation about photosynthesis
    # Ask follow-up question using "it" or "this process"
    # Verify system understands the context reference

def test_multi_source_integration():
    """Test system combines information from multiple curriculum sources."""
    # Ask broad question requiring multiple document sources
    # Verify response integrates information coherently
    # Check that all relevant sources are utilized
```

### test_response.py - Response Quality Testing
**Purpose**: Verify responses are formatted, personalized, and delivered appropriately

**What It Tests**:
- **Response Formatting** - Outputs are well-structured and readable
- **Personalization** - Responses adapted for individual users
- **Multilingual Support** - Proper English and Kinyarwanda response generation
- **Interactive Elements** - Follow-up questions and suggestions work correctly
- **Delivery Optimization** - Responses formatted correctly for different interfaces

**Test Examples**:
```python
def test_response_personalization():
    """Test responses are personalized for individual students."""
    # Create student profiles with different preferences
    # Ask same question for different students
    # Verify responses reflect individual preferences and history

def test_bilingual_responses():
    """Test system generates appropriate bilingual responses."""
    # Request explanation in Kinyarwanda
    # Verify response uses correct language
    # Check that technical terms are properly translated

def test_follow_up_generation():
    """Test system generates relevant follow-up questions."""
    # Provide explanation response
    # Verify follow-up questions are educational and relevant
    # Check questions encourage deeper learning
```

## For Contributors

### Implementation Status
This testing system is **comprehensively planned** with:

✅ **Complete Test Coverage** - All major system components have corresponding tests
🎓 **Educational Focus** - Tests validate pedagogical quality, not just technical function
🔍 **Quality Assurance** - Tests catch both technical bugs and educational issues
⚡ **Performance Validation** - Tests ensure system performs well under realistic load
🛠️ **Implementation Guides** - Clear examples of what and how to test

### Getting Started
1. **Start with `test_data_loader.py`** - Basic file processing tests
2. **Add `test_response.py`** - Response formatting and quality tests
3. **Implement `test_embeddings.py`** - Vector processing validation
4. **Build `test_models.py`** - AI model behavior verification
5. **Complete `test_rag.py`** - Core intelligence testing (most complex)

### Testing Philosophy

**Educational Quality First**:
- Tests verify educational value, not just technical functionality
- Responses must be pedagogically sound and age-appropriate
- Cultural sensitivity is validated alongside technical correctness

**Real-World Scenarios**:
- Tests use actual curriculum content and realistic questions
- Student interactions are modeled based on real learning patterns
- System performance tested under realistic usage conditions

**Comprehensive Coverage**:
- Unit tests for individual functions
- Integration tests for component interaction
- End-to-end tests for complete user workflows
- Performance tests for scalability validation

### Types of Tests You'll Write

**Functional Tests**:
```python
def test_quiz_generation():
    """Test that AI generates valid quiz questions from curriculum."""
    # Provide curriculum content on specific topic
    # Generate quiz questions
    # Verify questions test key concepts
    # Check answer choices are appropriate
    # Validate educational quality of questions
```

**Educational Quality Tests**:
```python
def test_explanation_clarity():
    """Test that AI explanations are clear and understandable."""
    # Generate explanation for complex topic
    # Verify language is appropriate for target grade
    # Check that examples are relevant and helpful
    # Validate logical flow and structure
```

**Cultural Appropriateness Tests**:
```python
def test_cultural_examples():
    """Test that AI uses appropriate Rwandan cultural context."""
    # Request examples for abstract concepts
    # Verify examples are from Rwandan context
    # Check cultural sensitivity and accuracy
    # Validate that examples aid understanding
```

**Performance Tests**:
```python
def test_response_time():
    """Test that system responds quickly enough for interactive use."""
    # Send typical student questions
    # Measure response times
    # Verify responses come back within 3-5 seconds
    # Test performance under multiple concurrent users
```

### Real-World Test Scenarios

**Student Learning Journey Test**:
```python
def test_adaptive_learning_progression():
    """Test system adapts to student learning over time."""
    1. Simulate student struggling with fractions
    2. Verify system provides simpler explanations
    3. Test that follow-up questions assess understanding
    4. Check system adjusts difficulty based on responses
    5. Validate learning progress is tracked accurately
```

**Teacher Content Management Test**:
```python
def test_curriculum_upload_workflow():
    """Test complete workflow of teacher uploading content."""
    1. Upload curriculum documents in various formats
    2. Verify content is processed and indexed
    3. Test that content appears in search results
    4. Check AI can use content in responses
    5. Validate content organization and metadata
```

### Testing Best Practices

**Test Data Management**:
- Use realistic Rwanda curriculum content for tests
- Include both English and Kinyarwanda test materials
- Create test data that represents various subjects and grade levels
- Maintain test data that doesn't change unexpectedly

**Automated Testing**:
- Tests run automatically when code changes
- Continuous integration ensures quality is maintained
- Performance tests run regularly to catch degradation
- Educational quality tests validate AI response appropriateness

**Test Documentation**:
- Each test clearly explains what it validates
- Test failures provide helpful debugging information
- Test coverage reports show which parts need more testing
- Educational rationale documented for quality tests

### Running Tests

**Development Testing**:
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_rag.py

# Run tests with coverage report
pytest --cov=app tests/
```

**Quality Assurance**:
- Tests must pass before code can be deployed
- Educational quality tests reviewed by education experts
- Performance tests validate system can handle expected usage
- Security tests ensure student data is protected

This testing folder ensures that the Rwanda AI Curriculum RAG system is not just technically sound, but educationally excellent and culturally appropriate!