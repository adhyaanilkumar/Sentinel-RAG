# Prompt Engineering Framework

This directory contains the prompt templates used by Sentinel RAG for tactical intelligence analysis.

## Design Philosophy

1. **Structured Format**: All prompts follow a consistent structure with ROLE, CONTEXT, TASK, CONSTRAINTS, and OUTPUT FORMAT sections.

2. **Version Control**: Prompts are version-controlled like code, with changes documented in CHANGELOG.md.

3. **Domain-Specific**: Prompts are tailored for military intelligence analysis, using appropriate terminology and formats.

4. **Reproducibility**: Prompts are stored as text files to ensure consistent behavior across runs.

## Prompt Files

| File | Purpose | Version |
|------|---------|---------|
| `image_analysis.txt` | GPT-4V image analysis for sensor data | 1.0 |
| `retrieval_query.txt` | Generate queries for RAG retrieval | 1.0 |
| `tactical_assessment.txt` | Final synthesis of intelligence | 1.0 |
| `follow_up.txt` | Chat follow-up responses | 1.0 |

## Usage

Prompts are loaded by the respective services (`VisionProcessor`, `LLMService`) at initialization.
Variables are substituted using Python's `.format()` method.

## Modifying Prompts

1. Make changes to the prompt file
2. Update the version number in the prompt header
3. Document changes in CHANGELOG.md
4. Test with evaluation dataset
5. Commit changes

## Prompt Template Format

```
# Prompt: [Name]
# Version: X.Y
# Last Updated: YYYY-MM-DD
# Purpose: [Brief description]

## ROLE
[Define the AI's role/persona]

## CONTEXT
{context_placeholder}

## TASK
[Specific instructions]

## CONSTRAINTS
[Limitations and guidelines]

## OUTPUT FORMAT
[Expected output structure]
```
