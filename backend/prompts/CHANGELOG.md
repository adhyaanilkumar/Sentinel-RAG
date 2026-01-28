# Prompt Iteration Log

This file documents the evolution of prompt templates used in Sentinel RAG.

## image_analysis.txt

### v1.0 (2024-01-20)
- Initial version
- Structured format with sensor type, objects, patterns, anomalies
- Includes military terminology instruction
- Emphasizes observation vs. inference distinction

## retrieval_query.txt

### v1.0 (2024-01-20)
- Initial version
- Generates concise search queries from image analysis
- Focuses on actionable keywords

## tactical_assessment.txt

### v1.0 (2024-01-20)
- Initial version
- JSON output format for structured parsing
- Includes threat level, intent analysis, recommendations
- Requires confidence rating and caveats

## follow_up.txt

### v1.0 (2024-01-20)
- Initial version
- Conversational follow-up for additional questions
- Maintains context from previous analysis

---

## Planned Improvements

- [ ] Add few-shot examples for better consistency
- [ ] Implement domain-specific terminology glossary
- [ ] Test temperature variations for assessment diversity
- [ ] Add chain-of-thought reasoning prompts
