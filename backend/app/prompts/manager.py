"""
PromptManager - Central management of prompt templates.

Handles loading, caching, and rendering of prompts with version tracking.
"""

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.prompts.template import PromptTemplate, PromptMetadata

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Centralized prompt template management.
    
    Features:
    - Load prompts from files or strings
    - Cache loaded templates
    - Track versions and changes
    - Provide rendering with variable substitution
    - Support hot-reload for development
    """
    
    def __init__(
        self,
        prompts_dir: str = "prompts",
        enable_cache: bool = True,
        auto_reload: bool = False,
    ):
        """
        Initialize the prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt template files
            enable_cache: Whether to cache loaded templates
            auto_reload: Whether to check for file changes on each access
        """
        self.prompts_dir = Path(prompts_dir)
        self.enable_cache = enable_cache
        self.auto_reload = auto_reload
        
        # Template cache: name -> (template, file_mtime, content_hash)
        self._cache: dict[str, tuple[PromptTemplate, float, str]] = {}
        
        # Load history for iteration tracking
        self._load_history: list[dict] = []
    
    def _get_file_path(self, name: str) -> Path:
        """Get the file path for a template name."""
        if not name.endswith(".txt"):
            name = f"{name}.txt"
        return self.prompts_dir / name
    
    def _compute_hash(self, content: str) -> str:
        """Compute hash of content for change detection."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _should_reload(self, name: str) -> bool:
        """Check if a cached template should be reloaded."""
        if not self.auto_reload or name not in self._cache:
            return name not in self._cache
        
        file_path = self._get_file_path(name)
        if not file_path.exists():
            return False
        
        _, cached_mtime, _ = self._cache[name]
        current_mtime = file_path.stat().st_mtime
        
        return current_mtime > cached_mtime
    
    def load(self, name: str, force_reload: bool = False) -> PromptTemplate:
        """
        Load a prompt template by name.
        
        Args:
            name: Template name (without .txt extension)
            force_reload: Force reload from file even if cached
            
        Returns:
            Loaded PromptTemplate
            
        Raises:
            FileNotFoundError: If template file doesn't exist
        """
        # Check cache
        if self.enable_cache and not force_reload and not self._should_reload(name):
            if name in self._cache:
                return self._cache[name][0]
        
        # Load from file
        file_path = self._get_file_path(name)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {file_path}")
        
        content = file_path.read_text(encoding="utf-8")
        content_hash = self._compute_hash(content)
        file_mtime = file_path.stat().st_mtime
        
        # Create template
        template = PromptTemplate(name=name, content=content)
        
        # Cache
        if self.enable_cache:
            self._cache[name] = (template, file_mtime, content_hash)
        
        # Log load event
        self._load_history.append({
            "name": name,
            "version": template.version,
            "timestamp": datetime.now().isoformat(),
            "content_hash": content_hash,
        })
        
        logger.debug(f"Loaded prompt template: {name} v{template.version}")
        return template
    
    def load_or_default(self, name: str, default_content: str) -> PromptTemplate:
        """
        Load a prompt template, falling back to default if not found.
        
        Args:
            name: Template name
            default_content: Default content if file doesn't exist
            
        Returns:
            Loaded or default PromptTemplate
        """
        try:
            return self.load(name)
        except FileNotFoundError:
            logger.warning(f"Prompt template {name} not found, using default")
            return PromptTemplate(name=name, content=default_content)
    
    def render(self, name: str, **variables) -> str:
        """
        Load and render a prompt template with variables.
        
        Args:
            name: Template name
            **variables: Variable values for substitution
            
        Returns:
            Rendered prompt string
        """
        template = self.load(name)
        return template.render(**variables)
    
    def render_safe(self, name: str, **variables) -> tuple[str, list[str]]:
        """
        Load and render a prompt with graceful error handling.
        
        Returns:
            Tuple of (rendered string, list of warnings)
        """
        try:
            template = self.load(name)
            return template.render_safe(**variables)
        except FileNotFoundError as e:
            return f"[PROMPT NOT FOUND: {name}]", [str(e)]
    
    def get_all_templates(self) -> list[PromptTemplate]:
        """Get all available prompt templates."""
        templates = []
        
        if not self.prompts_dir.exists():
            return templates
        
        for file_path in self.prompts_dir.glob("*.txt"):
            try:
                template = self.load(file_path.stem)
                templates.append(template)
            except Exception as e:
                logger.error(f"Error loading template {file_path}: {e}")
        
        return templates
    
    def get_template_info(self) -> list[dict]:
        """Get information about all available templates."""
        templates = self.get_all_templates()
        return [t.to_dict() for t in templates]
    
    def validate_all(self) -> dict[str, list[str]]:
        """
        Validate all templates for structural issues.
        
        Returns:
            Dictionary mapping template names to lists of issues
        """
        issues = {}
        
        for template in self.get_all_templates():
            template_issues = []
            
            # Check metadata
            if template.metadata.version == "Unknown":
                template_issues.append("Missing version in header")
            if template.metadata.purpose == "Unknown":
                template_issues.append("Missing purpose in header")
            
            # Check for unclosed braces
            content = template.content
            open_braces = content.count("{") - content.count("{{")
            close_braces = content.count("}") - content.count("}}")
            if open_braces != close_braces:
                template_issues.append(f"Mismatched braces: {open_braces} open, {close_braces} close")
            
            # Check for empty required variables
            if not template.required_variables and "{" in content:
                template_issues.append("Has placeholders but no required variables detected")
            
            if template_issues:
                issues[template.name] = template_issues
        
        return issues
    
    def get_load_history(self) -> list[dict]:
        """Get the history of template loads."""
        return self._load_history.copy()
    
    def clear_cache(self):
        """Clear the template cache."""
        self._cache.clear()
        logger.info("Prompt template cache cleared")
    
    def reload_all(self):
        """Force reload all cached templates."""
        names = list(self._cache.keys())
        self.clear_cache()
        
        for name in names:
            try:
                self.load(name, force_reload=True)
            except FileNotFoundError:
                logger.warning(f"Template {name} no longer exists")
    
    def compare_versions(self, name: str, other_content: str) -> dict:
        """
        Compare current template with another version.
        
        Args:
            name: Template name
            other_content: Content to compare against
            
        Returns:
            Comparison results
        """
        try:
            current = self.load(name)
            current_hash = self._compute_hash(current.content)
            other_hash = self._compute_hash(other_content)
            
            return {
                "name": name,
                "current_version": current.version,
                "hashes_match": current_hash == other_hash,
                "current_hash": current_hash,
                "other_hash": other_hash,
                "current_length": len(current.content),
                "other_length": len(other_content),
            }
        except FileNotFoundError:
            return {
                "name": name,
                "error": "Template not found",
            }


# Singleton instance for application use
_manager: Optional[PromptManager] = None


def get_manager(prompts_dir: str = "prompts") -> PromptManager:
    """Get or create the singleton PromptManager instance."""
    global _manager
    if _manager is None:
        _manager = PromptManager(prompts_dir=prompts_dir)
    return _manager


def reset_manager():
    """Reset the singleton manager (useful for testing)."""
    global _manager
    _manager = None
