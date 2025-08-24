# Configuration Folder - System Settings Management

This folder manages all the configuration settings for the Rwanda AI Curriculum RAG system across different environments (development, testing, production).

## What This Folder Does

Think of this as the **"control panel"** for your application. It handles:
- **Environment Settings** - Different configurations for development vs production
- **Secret Management** - Secure handling of passwords, API keys, and sensitive data
- **Application Constants** - Fixed values used throughout the system
- **Feature Flags** - Enable/disable features without code changes

## Folder Structure

```
config/
├── __init__.py           # Package initialization
├── settings.py           # Main application settings and environment configuration
├── secrets.py           # Secure secrets management (passwords, API keys, tokens)
└── constants.py         # Application constants and fixed values
```

## Files Explained

### settings.py - Application Settings
**Purpose**: Manages different configurations based on environment
- **Environment Detection** - Automatically detect if running in development/production
- **Database Settings** - Connection strings and database configuration
- **API Configuration** - Host, port, CORS settings
- **Feature Settings** - Enable/disable features like debugging, logging levels
- **Performance Settings** - Timeouts, limits, caching configuration

**Key Settings Include**:
- Server host and port (where the API runs)
- Database connection information
- Logging levels and file locations
- CORS (Cross-Origin Resource Sharing) settings
- Rate limiting configuration
- File upload limits and allowed types

### secrets.py - Secure Secrets Management
**Purpose**: Safely handle sensitive information that shouldn't be in code
- **Environment Variables** - Read secrets from system environment
- **Encryption Keys** - Keys for encrypting user data
- **API Keys** - External service authentication (OpenAI, etc.)
- **Database Passwords** - Secure database connection credentials
- **JWT Secrets** - Keys for creating and validating user tokens

**Important Security Features**:
- Never stores secrets directly in code files
- Reads from environment variables or secure vaults
- Provides fallback defaults for development
- Logs warnings when using insecure defaults
- Supports multiple secret sources (env vars, files, cloud vaults)

### constants.py - Application Constants
**Purpose**: Define fixed values used throughout the application
- **Supported Languages** - Available languages (English, Kinyarwanda)
- **User Roles** - Available user types (student, teacher, admin, etc.)
- **File Types** - Supported curriculum document formats
- **Grade Levels** - Rwandan education system grade levels
- **Subject Categories** - Available curriculum subjects

**Examples of Constants**:
```python
SUPPORTED_LANGUAGES = ["en", "rw"]
USER_ROLES = ["student", "teacher", "admin", "content_creator"]
SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".txt", ".md"]
GRADE_LEVELS = ["P1", "P2", "P3", "P4", "P5", "P6", "S1", "S2", "S3", "S4", "S5", "S6"]
```

## For Contributors

### Implementation Status
This configuration system is **fully designed** with:

✅ **Environment Management** - Automatic detection and switching
🔐 **Security Best Practices** - Safe secrets handling patterns
📋 **Comprehensive Settings** - All major configuration areas covered
🔧 **Easy Customization** - Simple to add new settings
🏗️ **Implementation Guides** - Clear instructions in each file

### Getting Started
1. **Start with `settings.py`** - Set up basic application configuration
2. **Configure `secrets.py`** - Set up secure environment variable handling
3. **Define `constants.py`** - Add fixed values your application needs
4. **Set environment variables** - Create `.env` file for local development

### Environment Setup
The system supports multiple environments:

**Development Environment**:
- Debug mode enabled
- Verbose logging
- Development database
- Hot reload enabled
- Test data and mock services

**Production Environment**:
- Security optimizations
- Error logging only
- Production database
- Performance optimizations
- Real external services

**Testing Environment**:
- Test database
- Mock external services
- Predictable test data
- Fast execution settings

### Configuration Examples

**Environment Variables (create a `.env` file)**:
```bash
# Database
DATABASE_URL=sqlite:///./test.db
DATABASE_PASSWORD=your_secure_password

# AI Services  
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_API_KEY=your_huggingface_key

# Security
JWT_SECRET_KEY=your_very_secure_random_key
ENCRYPTION_KEY=another_secure_random_key

# Application
ENVIRONMENT=development
HOST=127.0.0.1
PORT=8000
DEBUG=true
```

### Security Best Practices Included

1. **Never Commit Secrets** - All sensitive data comes from environment variables
2. **Secure Defaults** - Safe fallback values for development
3. **Validation** - Check that required secrets are provided
4. **Logging Safety** - Never log sensitive information
5. **Environment Separation** - Different settings for different environments

### Adding New Configuration

**To Add a New Setting**:
1. **Add to `settings.py`** - Define the setting with type hints
2. **Document the setting** - Explain what it does and valid values
3. **Set environment variable** - Add to your `.env` file
4. **Update documentation** - Add to README or config docs
5. **Use in your code** - Import and use the setting

**Example Adding a New Feature Flag**:
```python
# In settings.py
ENABLE_ADVANCED_ANALYTICS: bool = False

# In your .env file
ENABLE_ADVANCED_ANALYTICS=true

# In your application code
from app.config.settings import get_settings
settings = get_settings()
if settings.ENABLE_ADVANCED_ANALYTICS:
    # Advanced analytics code here
    pass
```

This configuration system makes your application flexible, secure, and easy to deploy in different environments!