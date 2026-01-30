"""
PromptRegistry - Registry of all prompt templates with metadata.

Provides a central catalog of available prompts for documentation and tooling.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.prompts.template import PromptTemplate
from app.prompts.manager import PromptManager, get_manager


class PromptCategory(Enum):
    """Categories for organizing prompts."""
    ANALYSIS = "analysis"
    RETRIEVAL = "retrieval"
    SYNTHESIS = "synthesis"
    CHAT = "chat"
    SYSTEM = "system"
    ERROR = "error"


@dataclass
class PromptRegistryEntry:
    """
    Registry entry for a prompt template.
    
    Contains metadata beyond what's in the template itself,
    useful for documentation and prompt selection.
    """
    name: str
    category: PromptCategory
    description: str
    file_name: str
    expected_output: str
    use_cases: list[str] = field(default_factory=list)
    model_requirements: Optional[str] = None
    temperature_recommendation: float = 0.0
    max_tokens_recommendation: int = 1024
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "file_name": self.file_name,
            "expected_output": self.expected_output,
            "use_cases": self.use_cases,
            "model_requirements": self.model_requirements,
            "temperature_recommendation": self.temperature_recommendation,
            "max_tokens_recommendation": self.max_tokens_recommendation,
        }


# Registry of all known prompts
PROMPT_REGISTRY: dict[str, PromptRegistryEntry] = {
    "image_analysis": PromptRegistryEntry(
        name="image_analysis",
        category=PromptCategory.ANALYSIS,
        description="Analyze sensor imagery (radar, sonar, satellite) to extract tactical information",
        file_name="image_analysis.txt",
        expected_output="Structured markdown with sections for sensor type, objects, patterns, anomalies",
        use_cases=[
            "Initial image processing",
            "Sensor data interpretation",
            "Object detection and classification",
        ],
        model_requirements="GPT-4V or equivalent vision model",
        temperature_recommendation=0.0,
        max_tokens_recommendation=1024,
    ),
    
    "retrieval_query": PromptRegistryEntry(
        name="retrieval_query",
        category=PromptCategory.RETRIEVAL,
        description="Generate search queries for knowledge base retrieval based on image analysis",
        file_name="retrieval_query.txt",
        expected_output="2-3 concise search queries",
        use_cases=[
            "RAG query generation",
            "Context retrieval preparation",
        ],
        model_requirements="GPT-4 or equivalent text model",
        temperature_recommendation=0.3,
        max_tokens_recommendation=256,
    ),
    
    "tactical_assessment": PromptRegistryEntry(
        name="tactical_assessment",
        category=PromptCategory.SYNTHESIS,
        description="Synthesize image analysis and historical context into actionable intelligence",
        file_name="tactical_assessment.txt",
        expected_output="JSON object with threat_level, observations, intent_analysis, recommendations",
        use_cases=[
            "Final intelligence synthesis",
            "Threat assessment generation",
            "Command briefing preparation",
        ],
        model_requirements="GPT-4 or equivalent text model",
        temperature_recommendation=0.0,
        max_tokens_recommendation=1024,
    ),
    
    "follow_up": PromptRegistryEntry(
        name="follow_up",
        category=PromptCategory.CHAT,
        description="Answer follow-up questions about tactical assessments",
        file_name="follow_up.txt",
        expected_output="Conversational response with context-aware information",
        use_cases=[
            "Chat interaction",
            "Clarification requests",
            "Deep-dive questions",
        ],
        model_requirements="GPT-4 or equivalent text model",
        temperature_recommendation=0.3,
        max_tokens_recommendation=512,
    ),
    
    "disambiguation": PromptRegistryEntry(
        name="disambiguation",
        category=PromptCategory.SYSTEM,
        description="Clarify ambiguous user queries or sensor data",
        file_name="disambiguation.txt",
        expected_output="Clarifying questions or interpretation options",
        use_cases=[
            "Handling unclear requests",
            "Multiple interpretation scenarios",
        ],
        model_requirements="GPT-4 or equivalent text model",
        temperature_recommendation=0.3,
        max_tokens_recommendation=256,
    ),
    
    "error_fallback": PromptRegistryEntry(
        name="error_fallback",
        category=PromptCategory.ERROR,
        description="Generate graceful responses when analysis fails or is unavailable",
        file_name="error_fallback.txt",
        expected_output="Helpful error message with next steps",
        use_cases=[
            "API failures",
            "Degraded mode operation",
            "Unsupported inputs",
        ],
        model_requirements=None,  # Can work with any model
        temperature_recommendation=0.0,
        max_tokens_recommendation=256,
    ),
    
    "system_context": PromptRegistryEntry(
        name="system_context",
        category=PromptCategory.SYSTEM,
        description="Establish system-wide context and behavioral guidelines",
        file_name="system_context.txt",
        expected_output="Contextual framing for all subsequent interactions",
        use_cases=[
            "Session initialization",
            "System identity establishment",
            "Behavioral guideline enforcement",
        ],
        model_requirements="GPT-4 or equivalent",
        temperature_recommendation=0.0,
        max_tokens_recommendation=512,
    ),
}


class PromptRegistry:
    """
    Registry providing access to prompt templates with rich metadata.
    
    Combines the PromptManager for loading with registry metadata
    for documentation and appropriate prompt selection.
    """
    
    def __init__(self, manager: Optional[PromptManager] = None):
        """
        Initialize the registry.
        
        Args:
            manager: PromptManager instance (uses singleton if not provided)
        """
        self.manager = manager or get_manager()
        self._registry = PROMPT_REGISTRY.copy()
    
    def get(self, name: str) -> Optional[PromptRegistryEntry]:
        """Get registry entry by name."""
        return self._registry.get(name)
    
    def get_template(self, name: str) -> PromptTemplate:
        """
        Get a prompt template by name.
        
        Args:
            name: Template name
            
        Returns:
            PromptTemplate instance
        """
        return self.manager.load(name)
    
    def get_by_category(self, category: PromptCategory) -> list[PromptRegistryEntry]:
        """Get all prompts in a category."""
        return [
            entry for entry in self._registry.values()
            if entry.category == category
        ]
    
    def list_all(self) -> list[str]:
        """List all registered prompt names."""
        return list(self._registry.keys())
    
    def get_all_entries(self) -> list[PromptRegistryEntry]:
        """Get all registry entries."""
        return list(self._registry.values())
    
    def render(self, name: str, **variables) -> str:
        """
        Render a prompt template by name.
        
        Uses the registry entry's recommended settings.
        """
        return self.manager.render(name, **variables)
    
    def get_recommended_params(self, name: str) -> dict:
        """
        Get recommended API parameters for a prompt.
        
        Returns:
            Dict with temperature, max_tokens, etc.
        """
        entry = self.get(name)
        if not entry:
            return {
                "temperature": 0.0,
                "max_tokens": 1024,
            }
        
        return {
            "temperature": entry.temperature_recommendation,
            "max_tokens": entry.max_tokens_recommendation,
            "model_requirements": entry.model_requirements,
        }
    
    def register(self, entry: PromptRegistryEntry):
        """
        Register a new prompt entry.
        
        Args:
            entry: PromptRegistryEntry to add
        """
        self._registry[entry.name] = entry
    
    def to_documentation(self) -> str:
        """Generate markdown documentation for all prompts."""
        lines = [
            "# Prompt Registry Documentation",
            "",
            "This document provides an overview of all registered prompt templates.",
            "",
        ]
        
        # Group by category
        for category in PromptCategory:
            entries = self.get_by_category(category)
            if not entries:
                continue
            
            lines.append(f"## {category.value.title()} Prompts")
            lines.append("")
            
            for entry in entries:
                lines.append(f"### {entry.name}")
                lines.append(f"**File:** `{entry.file_name}`")
                lines.append(f"**Description:** {entry.description}")
                lines.append(f"**Expected Output:** {entry.expected_output}")
                lines.append("")
                
                if entry.use_cases:
                    lines.append("**Use Cases:**")
                    for use_case in entry.use_cases:
                        lines.append(f"- {use_case}")
                    lines.append("")
                
                lines.append(f"**Recommended Settings:**")
                lines.append(f"- Temperature: {entry.temperature_recommendation}")
                lines.append(f"- Max Tokens: {entry.max_tokens_recommendation}")
                if entry.model_requirements:
                    lines.append(f"- Model: {entry.model_requirements}")
                lines.append("")
        
        return "\n".join(lines)


# Singleton instance
_registry: Optional[PromptRegistry] = None


def get_registry() -> PromptRegistry:
    """Get or create the singleton PromptRegistry instance."""
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry


def reset_registry():
    """Reset the singleton registry (useful for testing)."""
    global _registry
    _registry = None
