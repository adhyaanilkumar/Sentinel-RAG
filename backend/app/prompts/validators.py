"""
Prompt Validators - Validation utilities for prompt templates.

Provides functions to validate prompt structure, variables, and output format.
"""

import re
from typing import Optional

from app.prompts.template import PromptTemplate


class PromptValidationError(Exception):
    """Exception raised when prompt validation fails."""
    pass


def validate_header(content: str) -> tuple[bool, list[str]]:
    """
    Validate that a prompt has proper header metadata.
    
    Args:
        content: Raw prompt content
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    required_fields = [
        ("Prompt", r"#\s*Prompt:\s*\S+"),
        ("Version", r"#\s*Version:\s*\d+\.\d+"),
        ("Last Updated", r"#\s*Last Updated:\s*\d{4}-\d{2}-\d{2}"),
        ("Purpose", r"#\s*Purpose:\s*\S+"),
    ]
    
    for field_name, pattern in required_fields:
        if not re.search(pattern, content):
            issues.append(f"Missing or invalid header field: {field_name}")
    
    return (len(issues) == 0, issues)


def validate_structure(content: str) -> tuple[bool, list[str]]:
    """
    Validate that a prompt has proper section structure.
    
    Args:
        content: Raw prompt content
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    # Check for recommended sections
    recommended_sections = ["ROLE", "TASK", "OUTPUT FORMAT"]
    for section in recommended_sections:
        if f"## {section}" not in content and f"### {section}" not in content:
            issues.append(f"Missing recommended section: {section}")
    
    return (len(issues) == 0, issues)


def validate_variables(content: str, expected_vars: Optional[set[str]] = None) -> tuple[bool, list[str]]:
    """
    Validate variable placeholders in a prompt.
    
    Args:
        content: Raw prompt content
        expected_vars: Optional set of expected variable names
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    # Find all single-brace variables (excluding double-brace JSON examples)
    single_brace = set(re.findall(r'(?<!\{)\{(\w+)\}(?!\})', content))
    double_brace = set(re.findall(r'\{\{(\w+)\}\}', content))
    
    # Check for potential typos (single brace where double expected in JSON)
    json_keywords = {"sensor_type", "observations", "threat_level", "confidence"}
    for var in single_brace:
        if var in json_keywords and var in double_brace:
            issues.append(f"Variable '{var}' used inconsistently (both {{}} and {{{{}}}})")
    
    # Check expected variables
    if expected_vars:
        actual_vars = single_brace - double_brace
        missing = expected_vars - actual_vars
        extra = actual_vars - expected_vars
        
        for var in missing:
            issues.append(f"Expected variable not found: {var}")
        for var in extra:
            issues.append(f"Unexpected variable found: {var}")
    
    # Check for unmatched braces
    open_count = content.count("{")
    close_count = content.count("}")
    if open_count != close_count:
        issues.append(f"Unmatched braces: {open_count} open, {close_count} close")
    
    return (len(issues) == 0, issues)


def validate_output_format(content: str) -> tuple[bool, list[str]]:
    """
    Validate that output format section is well-defined.
    
    Args:
        content: Raw prompt content
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    # Find OUTPUT FORMAT section
    output_match = re.search(r"##\s*OUTPUT FORMAT\s*\n([\s\S]*?)(?=\n##|\Z)", content)
    
    if not output_match:
        issues.append("No OUTPUT FORMAT section found")
        return (False, issues)
    
    output_section = output_match.group(1)
    
    # Check if it has some specification
    if len(output_section.strip()) < 20:
        issues.append("OUTPUT FORMAT section is too brief")
    
    # Check for JSON format if it's supposed to output JSON
    if "json" in content.lower() and "```json" not in output_section.lower():
        # This is just a warning, not an error
        pass
    
    return (len(issues) == 0, issues)


def validate_constraints(content: str) -> tuple[bool, list[str]]:
    """
    Check that constraints section uses proper emphasis.
    
    Args:
        content: Raw prompt content
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    # Find CONSTRAINTS section
    constraints_match = re.search(r"##\s*CONSTRAINTS\s*\n([\s\S]*?)(?=\n##|\Z)", content)
    
    if not constraints_match:
        # Not required, just return valid
        return (True, [])
    
    constraints = constraints_match.group(1)
    
    # Check for emphasis keywords (should use CAPS or bold for important terms)
    important_terms = ["not", "must", "always", "never", "do not"]
    for term in important_terms:
        pattern = rf"\b{term}\b"
        if re.search(pattern, constraints, re.IGNORECASE):
            caps_pattern = rf"\b{term.upper()}\b"
            if not re.search(caps_pattern, constraints):
                issues.append(f"Consider emphasizing '{term}' in CONSTRAINTS section")
    
    return (len(issues) == 0, issues)


def validate_prompt(template: PromptTemplate, strict: bool = False) -> tuple[bool, list[str]]:
    """
    Comprehensive validation of a prompt template.
    
    Args:
        template: PromptTemplate to validate
        strict: If True, treat warnings as errors
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    all_issues = []
    
    # Header validation
    is_valid, issues = validate_header(template.content)
    if not is_valid:
        all_issues.extend(issues)
    
    # Structure validation
    is_valid, issues = validate_structure(template.content)
    if not is_valid and strict:
        all_issues.extend(issues)
    
    # Variable validation
    is_valid, issues = validate_variables(template.content)
    if not is_valid:
        all_issues.extend(issues)
    
    # Output format validation
    is_valid, issues = validate_output_format(template.content)
    if not is_valid:
        all_issues.extend(issues)
    
    # Constraints validation (warnings only)
    if strict:
        _, issues = validate_constraints(template.content)
        all_issues.extend(issues)
    
    return (len(all_issues) == 0, all_issues)


def validate_rendering(template: PromptTemplate, test_values: dict[str, str]) -> tuple[bool, str, list[str]]:
    """
    Test that a template renders correctly with test values.
    
    Args:
        template: PromptTemplate to test
        test_values: Test values for variables
        
    Returns:
        Tuple of (is_valid, rendered_output, list of issues)
    """
    issues = []
    
    try:
        rendered = template.render(**test_values)
        
        # Check for unsubstituted variables
        remaining_vars = re.findall(r'(?<!\{)\{(\w+)\}(?!\})', rendered)
        if remaining_vars:
            issues.append(f"Unsubstituted variables after rendering: {remaining_vars}")
        
        # Check output isn't empty
        if len(rendered.strip()) < 50:
            issues.append("Rendered output is suspiciously short")
        
        return (len(issues) == 0, rendered, issues)
        
    except ValueError as e:
        return (False, "", [str(e)])


class PromptLinter:
    """
    Linter for prompt templates that checks for common issues.
    """
    
    def __init__(self):
        self.rules = [
            self._check_length,
            self._check_clarity,
            self._check_specificity,
            self._check_examples,
        ]
    
    def _check_length(self, content: str) -> list[str]:
        """Check prompt isn't too long or too short."""
        issues = []
        word_count = len(content.split())
        
        if word_count < 50:
            issues.append(f"Prompt may be too short ({word_count} words) - consider adding more detail")
        elif word_count > 1500:
            issues.append(f"Prompt may be too long ({word_count} words) - consider condensing")
        
        return issues
    
    def _check_clarity(self, content: str) -> list[str]:
        """Check for clarity issues."""
        issues = []
        
        # Check for vague language
        vague_terms = ["etc", "and so on", "things like", "stuff"]
        for term in vague_terms:
            if term.lower() in content.lower():
                issues.append(f"Consider replacing vague term: '{term}'")
        
        return issues
    
    def _check_specificity(self, content: str) -> list[str]:
        """Check for specificity in instructions."""
        issues = []
        
        # Check for overly generic instructions
        generic_phrases = [
            "be helpful",
            "do your best",
            "try to",
        ]
        for phrase in generic_phrases:
            if phrase.lower() in content.lower():
                issues.append(f"Consider more specific instruction instead of: '{phrase}'")
        
        return issues
    
    def _check_examples(self, content: str) -> list[str]:
        """Check if examples might help."""
        issues = []
        
        # If output format is complex but no examples given
        if "json" in content.lower() or "format" in content.lower():
            if "example" not in content.lower() and "```" not in content:
                issues.append("Consider adding examples for complex output formats")
        
        return issues
    
    def lint(self, content: str) -> list[str]:
        """
        Run all linting rules on a prompt.
        
        Args:
            content: Prompt content to lint
            
        Returns:
            List of linting suggestions (not errors)
        """
        all_issues = []
        for rule in self.rules:
            issues = rule(content)
            all_issues.extend(issues)
        return all_issues
