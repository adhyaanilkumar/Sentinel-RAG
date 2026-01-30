# Prompt Engineering Framework

This directory contains the prompt templates used by Sentinel RAG for tactical intelligence analysis.

## Overview

The Sentinel RAG prompt engineering framework provides:

- **Structured Templates**: Consistent format with ROLE, CONTEXT, TASK, CONSTRAINTS, and OUTPUT FORMAT sections
- **Version Control**: All prompts are versioned with changes documented in `CHANGELOG.md`
- **Python Integration**: `app.prompts` module for template loading, rendering, and validation
- **Iteration Tracking**: Experiments logged in `experiments.json` for reproducibility

## Quick Start

```python
from app.prompts import PromptManager, get_registry

# Using PromptManager directly
manager = PromptManager(prompts_dir="prompts")
rendered = manager.render("tactical_assessment",
    image_analysis="...",
    context="...",
    additional_context="..."
)

# Using the Registry for metadata
registry = get_registry()
params = registry.get_recommended_params("tactical_assessment")
template = registry.get_template("tactical_assessment")
```

## Prompt Files

| File | Purpose | Category | Version |
|------|---------|----------|---------|
| `image_analysis.txt` | GPT-4V image analysis for sensor data | Analysis | 1.0 |
| `retrieval_query.txt` | Generate queries for RAG retrieval | Retrieval | 1.0 |
| `tactical_assessment.txt` | Final synthesis of intelligence | Synthesis | 1.0 |
| `follow_up.txt` | Chat follow-up responses | Chat | 1.0 |
| `disambiguation.txt` | Clarify ambiguous queries | System | 1.0 |
| `error_fallback.txt` | Graceful degradation responses | Error | 1.0 |
| `system_context.txt` | System identity and guidelines | System | 1.0 |

## Template Format

All prompts follow this structure:

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

### Variables

Variables use Python format string syntax: `{variable_name}`

For JSON examples within prompts, use double braces to escape: `{{key}}`

## Python Module (`app.prompts`)

### PromptTemplate

Represents a single prompt with metadata and variable handling:

```python
from app.prompts import PromptTemplate

template = PromptTemplate(name="test", content="Hello {name}!")
rendered = template.render(name="World")
# "Hello World!"

# Safe rendering with warnings
rendered, warnings = template.render_safe(name="World")
```

### PromptManager

Centralized template management with caching:

```python
from app.prompts import PromptManager

manager = PromptManager(
    prompts_dir="prompts",
    enable_cache=True,
    auto_reload=False  # Set True for development
)

# Load and render
template = manager.load("image_analysis")
result = manager.render("image_analysis")

# Validation
issues = manager.validate_all()
```

### PromptRegistry

Registry with metadata for prompt selection:

```python
from app.prompts import get_registry, PromptCategory

registry = get_registry()

# Get prompts by category
analysis_prompts = registry.get_by_category(PromptCategory.ANALYSIS)

# Get recommended parameters
params = registry.get_recommended_params("tactical_assessment")
# {"temperature": 0.0, "max_tokens": 1024, ...}

# Generate documentation
docs = registry.to_documentation()
```

### Iteration Tracking

Log and track prompt experiments:

```python
from app.prompts.iteration import IterationLog

log = IterationLog(prompts_dir="prompts")

# Log a change
log.log_change(
    prompt_name="image_analysis",
    version="1.1",
    change_type="refinement",
    description="Added few-shot examples",
    rationale="Improve output consistency",
    changes_made=["Added 2 example analyses"]
)

# Log test results
log.log_test_results(
    prompt_name="image_analysis",
    version="1.1",
    test_cases_run=10,
    test_cases_passed=9
)

# Log quality metrics
log.log_quality_metrics(
    prompt_name="image_analysis",
    version="1.1",
    relevance=0.92,
    coherence=0.88,
    format_compliance=0.95
)
```

### Validation

Validate prompts for structural issues:

```python
from app.prompts.validators import validate_prompt, PromptLinter

template = manager.load("tactical_assessment")

# Full validation
is_valid, issues = validate_prompt(template, strict=True)

# Linting for style suggestions
linter = PromptLinter()
suggestions = linter.lint(template.content)
```

## Modifying Prompts

1. **Make changes** to the prompt file
2. **Update version** number in the prompt header (e.g., 1.0 → 1.1)
3. **Log the change** using `IterationLog.log_change()`
4. **Run tests** with evaluation dataset
5. **Update CHANGELOG.md** (automatic if using IterationLog)
6. **Commit changes** with descriptive message

### Version Numbering

- **Major (X.0)**: Breaking changes to output format or significant restructuring
- **Minor (X.Y)**: Refinements, additions, or improvements

## A/B Testing

To compare prompt versions:

```python
from app.prompts.iteration import IterationLog

log = IterationLog()

# After testing both versions
experiment = log.log_change(
    prompt_name="image_analysis",
    version="1.1",
    change_type="refinement",
    description="Test improved clarity instructions",
    rationale="Hypothesis: clearer instructions improve format compliance"
)

# Record comparison
experiment.compared_with = "1.0"
experiment.improvement_notes = "Format compliance improved from 85% to 95%"
```

## Best Practices

1. **Be Specific**: Use precise language in instructions
2. **Show Examples**: Include example outputs for complex formats
3. **Constrain Clearly**: Use CAPS for emphasis in CONSTRAINTS section
4. **Test Thoroughly**: Run against evaluation dataset before deploying
5. **Document Changes**: Always update CHANGELOG.md with rationale
6. **Version Control**: Treat prompts like code - review changes carefully

## Files

```
prompts/
├── README.md              # This file
├── CHANGELOG.md           # Iteration history
├── experiments.json       # Structured experiment data
├── image_analysis.txt     # Vision analysis prompt
├── retrieval_query.txt    # RAG query generation
├── tactical_assessment.txt # Intelligence synthesis
├── follow_up.txt          # Chat responses
├── disambiguation.txt     # Ambiguity resolution
├── error_fallback.txt     # Graceful degradation
└── system_context.txt     # System identity
```
