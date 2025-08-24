# Prompts Folder - AI Instruction Templates

This folder contains carefully crafted prompt templates that guide AI models to generate high-quality educational content for the Rwanda AI Curriculum RAG system.

## What This Folder Does

This is the **"AI instruction manual"** that:
- **Guides AI Behavior** - Tell AI models exactly how to respond in educational contexts
- **Ensures Consistency** - Standardize AI responses across different features
- **Maintains Quality** - Use proven prompt patterns for best results
- **Cultural Adaptation** - Include Rwanda-specific context and examples
- **Educational Focus** - Optimize for learning and teaching scenarios

Think of prompts as the "scripts" that tell AI how to be an excellent teacher or tutor.

## Folder Structure

```
prompts/
├── __init__.py              # Package initialization
├── learning_prompts.py      # Prompts for teaching and learning interactions
├── quiz_prompts.py         # Prompts for quiz and assessment generation
└── utils.py                # Prompt utilities and helper functions
```

## Files Explained

### learning_prompts.py - Educational Interaction Prompts
**Purpose**: Guide AI to be an effective teacher and learning assistant

**What It Contains**:
- **Tutoring Prompts** - Help AI explain concepts clearly and patiently
- **Explanation Templates** - Structure for breaking down complex topics
- **Conversation Starters** - Engage students in educational discussions
- **Difficulty Adaptation** - Adjust explanations for different grade levels
- **Cultural Integration** - Include Rwandan context and examples

**Key Prompt Categories**:

**Concept Explanation Prompts**:
```
"You are a patient Rwandan teacher explaining {topic} to a {grade_level} student. 
Use simple language, local examples from Rwanda, and check for understanding. 
If the concept is difficult, break it into smaller steps..."
```

**Socratic Learning Prompts**:
```
"Guide the student to discover the answer through questions. Don't give the answer directly. 
Ask leading questions that help them think through the problem step by step..."
```

**Cultural Context Prompts**:
```
"When explaining {subject}, use examples from Rwandan culture, geography, and daily life. 
Reference familiar places, foods, traditions, or situations that students can relate to..."
```

### quiz_prompts.py - Assessment Generation Prompts
**Purpose**: Guide AI to create high-quality educational assessments

**What It Contains**:
- **Question Generation** - Create various types of quiz questions
- **Answer Key Creation** - Generate comprehensive answer explanations
- **Difficulty Calibration** - Ensure appropriate challenge level
- **Assessment Rubrics** - Create grading criteria and feedback
- **Learning Objective Alignment** - Match questions to curriculum goals

**Key Prompt Categories**:

**Multiple Choice Generation**:
```
"Create a multiple choice question about {topic} for {grade_level} students. 
Include 4 options with one correct answer. Make distractors plausible but clearly incorrect. 
Use Rwandan context in examples where appropriate..."
```

**Critical Thinking Questions**:
```
"Generate an open-ended question that requires students to analyze, evaluate, or synthesize 
information about {topic}. The question should encourage deeper thinking beyond memorization..."
```

**Practical Application Questions**:
```
"Create a question that asks students to apply {concept} to a real-world situation in Rwanda. 
The scenario should be relevant to students' lives and demonstrate practical understanding..."
```

### utils.py - Prompt Management Utilities
**Purpose**: Tools for managing, formatting, and optimizing prompts

**What It Contains**:
- **Template Management** - Load and format prompt templates
- **Dynamic Content Insertion** - Insert specific topics, grade levels, etc.
- **Prompt Validation** - Check that prompts produce expected results
- **A/B Testing Tools** - Compare different prompt variations
- **Multilingual Support** - Handle English and Kinyarwanda prompts

**Utility Functions**:
- **Prompt Formatting** - Insert variables into templates
- **Context Building** - Add relevant background information
- **Response Parsing** - Extract structured information from AI responses
- **Quality Checking** - Validate AI outputs meet educational standards
- **Localization** - Adapt prompts for Rwandan cultural context

## For Contributors

### Implementation Status
This prompt system is **expertly designed** with:

✅ **Educational Best Practices** - Based on proven teaching methods
🇷🇼 **Rwanda-Specific Context** - Culturally appropriate examples and references
🎓 **Pedagogically Sound** - Aligned with learning science principles
📚 **Comprehensive Coverage** - Prompts for all major educational interactions
🛠️ **Implementation Guides** - Clear examples and usage instructions

### Getting Started
1. **Start with `learning_prompts.py`** - Basic educational interaction templates
2. **Add `quiz_prompts.py`** - Assessment generation prompts
3. **Implement `utils.py`** - Prompt management tools
4. **Test with AI models** - Verify prompts produce good educational content
5. **Refine based on results** - Improve prompts based on AI outputs

### Prompt Engineering Best Practices

**Clear Instructions**:
```python
# Good prompt - specific and clear
"You are a friendly math tutor helping a Primary 4 student in Rwanda. 
Explain multiplication using examples of counting banana bunches or groups of students."

# Poor prompt - vague and general  
"Help with math."
```

**Educational Structure**:
```python
# Include pedagogical elements
EXPLANATION_TEMPLATE = """
1. Start with what the student already knows
2. Introduce the new concept with a simple definition
3. Provide a concrete example from Rwandan context
4. Check for understanding with a question
5. Offer practice opportunity
"""
```

**Cultural Sensitivity**:
```python
# Rwanda-appropriate examples
EXAMPLES = {
    "counting": "counting cows, banana plants, or students",
    "fractions": "sharing ubwoba (porridge) or dividing land",
    "geography": "rivers like Nyabarongo, mountains like Nyiragongo"
}
```

### Real-World Usage Examples

**Adaptive Learning Chat**:
```
Prompt: "Student asks about photosynthesis but seems confused. 
Previous responses suggest they understand plants need water and sunlight 
but struggle with the chemical process. Use agricultural examples from Rwanda..."

AI Response: "Think about the sweet potato plants you see growing. 
They take in water through their roots and sunlight through their leaves, 
just like we learned. But here's the amazing part - inside the leaves, 
the plant is actually making its own food..."
```

**Quiz Generation**:
```  
Prompt: "Generate 5 questions about Rwanda's geography for Senior 2 students. 
Include one map-reading question, one about climate, one about natural resources, 
one comparison question, and one application question..."

AI Output: Multiple relevant, well-structured questions with appropriate 
difficulty and local context.
```

### Key Features You'll Implement

**Dynamic Prompts**:
- **Variable Insertion** - Customize prompts with specific topics, grades, contexts
- **Conditional Logic** - Different prompts based on student performance or needs
- **Context Awareness** - Include relevant background from previous interactions
- **Personalization** - Adapt to individual student learning styles

**Quality Assurance**:
- **Educational Validation** - Ensure AI responses support learning objectives
- **Cultural Review** - Verify appropriate use of Rwandan context
- **Language Quality** - Check for clear, age-appropriate language
- **Safety Filtering** - Prevent inappropriate content in educational settings

**Multilingual Support**:
- **Kinyarwanda Integration** - Prompts that effectively use local language
- **Translation Guidance** - Help AI translate educational concepts accurately
- **Code-Switching** - Natural mixing of English and Kinyarwanda where appropriate
- **Cultural Bridge** - Connect universal concepts with local understanding

### Prompt Categories You'll Create

**Learning Assistance**:
- Concept explanation prompts
- Problem-solving guidance  
- Misconception correction
- Learning encouragement

**Assessment Creation**:
- Question generation templates
- Answer key development
- Rubric creation guides
- Feedback generation

**Content Adaptation**:
- Grade level adjustment
- Cultural localization
- Language simplification
- Example generation

### Testing Your Prompts
- **Educational Effectiveness** - Do AI responses help students learn?
- **Cultural Appropriateness** - Are examples relevant to Rwandan students?
- **Language Quality** - Is the language clear and age-appropriate?
- **Consistency** - Do similar prompts produce reliably good results?
- **Edge Cases** - How do prompts handle unusual or difficult questions?

This prompts folder is what transforms a generic AI model into a skilled, culturally-aware Rwandan educator!