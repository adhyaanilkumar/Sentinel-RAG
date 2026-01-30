"""
PromptTemplate - Structured prompt template with version tracking.

Each template maintains metadata about its purpose, version, and required variables.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class PromptMetadata:
    """Metadata extracted from a prompt template file."""
    name: str
    version: str
    last_updated: str
    purpose: str
    
    @classmethod
    def from_header(cls, content: str) -> Optional["PromptMetadata"]:
        """Parse metadata from prompt file header comments."""
        patterns = {
            "name": r"#\s*Prompt:\s*(.+)",
            "version": r"#\s*Version:\s*(.+)",
            "last_updated": r"#\s*Last Updated:\s*(.+)",
            "purpose": r"#\s*Purpose:\s*(.+)",
        }
        
        values = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            values[key] = match.group(1).strip() if match else "Unknown"
        
        return cls(**values)


@dataclass
class PromptTemplate:
    """
    A structured prompt template with version tracking and variable management.
    
    Attributes:
        name: Unique identifier for the template
        content: The raw template content with placeholders
        metadata: Parsed metadata from the template header
        required_variables: Set of variable names that must be provided
        optional_variables: Set of variable names with defaults
        defaults: Default values for optional variables
    """
    name: str
    content: str
    metadata: Optional[PromptMetadata] = None
    required_variables: set[str] = field(default_factory=set)
    optional_variables: set[str] = field(default_factory=set)
    defaults: dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Extract variables and metadata after initialization."""
        if self.metadata is None:
            self.metadata = PromptMetadata.from_header(self.content)
        
        # Extract all placeholders from content
        all_vars = set(re.findall(r'\{(\w+)\}', self.content))
        
        # Remove escaped double braces (JSON examples)
        double_brace_vars = set(re.findall(r'\{\{(\w+)\}\}', self.content))
        all_vars -= double_brace_vars
        
        # If not explicitly set, all found variables are required
        if not self.required_variables and not self.optional_variables:
            self.required_variables = all_vars
    
    @property
    def version(self) -> str:
        """Get the template version."""
        return self.metadata.version if self.metadata else "unknown"
    
    def get_variables(self) -> dict[str, str]:
        """Get all variables with their requirement status."""
        result = {}
        for var in self.required_variables:
            result[var] = "required"
        for var in self.optional_variables:
            result[var] = f"optional (default: {self.defaults.get(var, 'None')})"
        return result
    
    def validate_variables(self, provided: dict[str, str]) -> tuple[bool, list[str]]:
        """
        Validate that all required variables are provided.
        
        Args:
            provided: Dictionary of variable names to values
            
        Returns:
            Tuple of (is_valid, list of missing/invalid variables)
        """
        missing = []
        
        for var in self.required_variables:
            if var not in provided or provided[var] is None:
                missing.append(f"Missing required variable: {var}")
        
        return (len(missing) == 0, missing)
    
    def render(self, **variables) -> str:
        """
        Render the template with provided variables.
        
        Args:
            **variables: Variable values to substitute
            
        Returns:
            Rendered prompt string
            
        Raises:
            ValueError: If required variables are missing
        """
        # Merge defaults with provided variables
        merged = {**self.defaults, **variables}
        
        # Validate
        is_valid, errors = self.validate_variables(merged)
        if not is_valid:
            raise ValueError(f"Template validation failed: {'; '.join(errors)}")
        
        # Perform substitution
        # Use safe substitution that handles missing optional variables
        result = self.content
        for var_name, value in merged.items():
            if value is not None:
                result = result.replace(f"{{{var_name}}}", str(value))
        
        return result
    
    def render_safe(self, **variables) -> tuple[str, list[str]]:
        """
        Render template with graceful handling of missing variables.
        
        Returns:
            Tuple of (rendered string, list of warnings)
        """
        warnings = []
        merged = {**self.defaults, **variables}
        
        result = self.content
        for var_name in self.required_variables | self.optional_variables:
            placeholder = f"{{{var_name}}}"
            if var_name in merged and merged[var_name] is not None:
                result = result.replace(placeholder, str(merged[var_name]))
            elif var_name in self.required_variables:
                warnings.append(f"Missing required variable: {var_name}")
                result = result.replace(placeholder, f"[MISSING: {var_name}]")
            else:
                result = result.replace(placeholder, "")
        
        return result, warnings
    
    def to_dict(self) -> dict:
        """Serialize template to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "metadata": {
                "name": self.metadata.name,
                "version": self.metadata.version,
                "last_updated": self.metadata.last_updated,
                "purpose": self.metadata.purpose,
            } if self.metadata else None,
            "required_variables": list(self.required_variables),
            "optional_variables": list(self.optional_variables),
            "defaults": self.defaults,
        }
    
    def __repr__(self) -> str:
        return f"PromptTemplate(name='{self.name}', version='{self.version}')"


def create_template(
    name: str,
    content: str,
    required: Optional[list[str]] = None,
    optional: Optional[list[str]] = None,
    defaults: Optional[dict[str, str]] = None,
) -> PromptTemplate:
    """
    Factory function to create a PromptTemplate with explicit variable definitions.
    
    Args:
        name: Template name
        content: Template content
        required: List of required variable names
        optional: List of optional variable names
        defaults: Default values for optional variables
        
    Returns:
        Configured PromptTemplate instance
    """
    return PromptTemplate(
        name=name,
        content=content,
        required_variables=set(required) if required else set(),
        optional_variables=set(optional) if optional else set(),
        defaults=defaults or {},
    )
