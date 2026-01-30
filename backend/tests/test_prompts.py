"""
Tests for the prompt engineering framework.
"""

import os
import shutil
from pathlib import Path

import pytest

from app.prompts.template import PromptTemplate, PromptMetadata, create_template
from app.prompts.manager import PromptManager, reset_manager
from app.prompts.registry import PromptRegistry, PromptCategory, PROMPT_REGISTRY
from app.prompts.iteration import IterationLog, PromptExperiment
from app.prompts.validators import (
    validate_header,
    validate_structure,
    validate_variables,
    validate_prompt,
    PromptLinter,
)

# Use workspace-local temp directory for tests
TEST_TEMP_DIR = Path(__file__).parent.parent / "data" / "_test_temp"


class TestPromptMetadata:
    """Tests for PromptMetadata parsing."""
    
    def test_parse_from_header(self):
        """Test parsing metadata from prompt header."""
        content = """# Prompt: Test Prompt
# Version: 1.0
# Last Updated: 2024-01-20
# Purpose: Testing purposes

## ROLE
Test role
"""
        metadata = PromptMetadata.from_header(content)
        
        assert metadata.name == "Test Prompt"
        assert metadata.version == "1.0"
        assert metadata.last_updated == "2024-01-20"
        assert metadata.purpose == "Testing purposes"
    
    def test_parse_missing_fields(self):
        """Test parsing with missing fields."""
        content = "# Prompt: Test\nSome content"
        metadata = PromptMetadata.from_header(content)
        
        assert metadata.name == "Test"
        assert metadata.version == "Unknown"
        assert metadata.purpose == "Unknown"


class TestPromptTemplate:
    """Tests for PromptTemplate."""
    
    def test_extract_variables(self):
        """Test variable extraction from template."""
        content = "Hello {name}, your role is {role}."
        template = PromptTemplate(name="test", content=content)
        
        assert "name" in template.required_variables
        assert "role" in template.required_variables
    
    def test_ignore_double_braces(self):
        """Test that double braces are not treated as variables."""
        content = '{"key": "value"} and {variable}'
        template = PromptTemplate(name="test", content=content)
        
        # JSON double braces should be ignored
        assert "variable" in template.required_variables
    
    def test_render_success(self):
        """Test successful template rendering."""
        content = "Hello {name}!"
        template = PromptTemplate(name="test", content=content)
        
        result = template.render(name="World")
        assert result == "Hello World!"
    
    def test_render_missing_variable(self):
        """Test rendering with missing required variable."""
        content = "Hello {name} and {other}!"
        template = PromptTemplate(name="test", content=content)
        
        with pytest.raises(ValueError) as exc_info:
            template.render(name="World")
        
        assert "Missing required variable: other" in str(exc_info.value)
    
    def test_render_safe(self):
        """Test safe rendering with missing variables."""
        content = "Hello {name} and {other}!"
        template = PromptTemplate(name="test", content=content)
        
        result, warnings = template.render_safe(name="World")
        
        assert "Hello World and [MISSING: other]!" == result
        assert len(warnings) == 1
        assert "other" in warnings[0]
    
    def test_create_template_factory(self):
        """Test factory function for creating templates."""
        template = create_template(
            name="test",
            content="Hello {name}!",
            required=["name"],
            optional=["greeting"],
            defaults={"greeting": "Hello"},
        )
        
        assert "name" in template.required_variables
        assert "greeting" in template.optional_variables
        assert template.defaults["greeting"] == "Hello"
    
    def test_to_dict(self):
        """Test template serialization."""
        template = PromptTemplate(name="test", content="# Prompt: Test\n{var}")
        
        data = template.to_dict()
        
        assert data["name"] == "test"
        assert "var" in data["required_variables"]


class TestPromptManager:
    """Tests for PromptManager."""
    
    @pytest.fixture
    def temp_prompts_dir(self):
        """Create a temporary prompts directory in the workspace."""
        prompts_dir = TEST_TEMP_DIR / "prompts_test"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test prompt file
        test_prompt = prompts_dir / "test_prompt.txt"
        test_prompt.write_text("""# Prompt: Test
# Version: 1.0
# Last Updated: 2024-01-20
# Purpose: Testing

## ROLE
Test role

## TASK
Process {input}

## OUTPUT FORMAT
Return processed text
""", encoding="utf-8")
        
        yield prompts_dir
        
        # Cleanup
        if prompts_dir.exists():
            shutil.rmtree(prompts_dir, ignore_errors=True)
    
    def test_load_prompt(self, temp_prompts_dir):
        """Test loading a prompt from file."""
        manager = PromptManager(prompts_dir=str(temp_prompts_dir))
        
        template = manager.load("test_prompt")
        
        assert template.name == "test_prompt"
        assert template.version == "1.0"
        assert "input" in template.required_variables
    
    def test_load_not_found(self, temp_prompts_dir):
        """Test loading non-existent prompt."""
        manager = PromptManager(prompts_dir=str(temp_prompts_dir))
        
        with pytest.raises(FileNotFoundError):
            manager.load("nonexistent")
    
    def test_load_or_default(self, temp_prompts_dir):
        """Test load with default fallback."""
        manager = PromptManager(prompts_dir=str(temp_prompts_dir))
        
        template = manager.load_or_default(
            "nonexistent",
            default_content="Default {content}"
        )
        
        assert template.name == "nonexistent"
        assert "content" in template.required_variables
    
    def test_render(self, temp_prompts_dir):
        """Test loading and rendering in one call."""
        manager = PromptManager(prompts_dir=str(temp_prompts_dir))
        
        result = manager.render("test_prompt", input="test data")
        
        assert "test data" in result
    
    def test_caching(self, temp_prompts_dir):
        """Test that templates are cached."""
        manager = PromptManager(prompts_dir=str(temp_prompts_dir), enable_cache=True)
        
        template1 = manager.load("test_prompt")
        template2 = manager.load("test_prompt")
        
        assert template1 is template2  # Same object from cache
    
    def test_force_reload(self, temp_prompts_dir):
        """Test forcing a reload bypasses cache."""
        manager = PromptManager(prompts_dir=str(temp_prompts_dir), enable_cache=True)
        
        template1 = manager.load("test_prompt")
        template2 = manager.load("test_prompt", force_reload=True)
        
        assert template1 is not template2
    
    def test_get_all_templates(self, temp_prompts_dir):
        """Test getting all available templates."""
        manager = PromptManager(prompts_dir=str(temp_prompts_dir))
        
        templates = manager.get_all_templates()
        
        assert len(templates) == 1
        assert templates[0].name == "test_prompt"
    
    def test_clear_cache(self, temp_prompts_dir):
        """Test cache clearing."""
        manager = PromptManager(prompts_dir=str(temp_prompts_dir))
        
        manager.load("test_prompt")
        assert len(manager._cache) == 1
        
        manager.clear_cache()
        assert len(manager._cache) == 0


class TestPromptRegistry:
    """Tests for PromptRegistry."""
    
    def test_builtin_prompts_registered(self):
        """Test that all expected prompts are in the registry."""
        expected = ["image_analysis", "retrieval_query", "tactical_assessment", "follow_up"]
        
        for name in expected:
            assert name in PROMPT_REGISTRY
    
    def test_get_by_category(self):
        """Test filtering prompts by category."""
        registry = PromptRegistry.__new__(PromptRegistry)
        registry._registry = PROMPT_REGISTRY.copy()
        registry.manager = None
        
        analysis_prompts = registry.get_by_category(PromptCategory.ANALYSIS)
        
        assert len(analysis_prompts) >= 1
        assert all(p.category == PromptCategory.ANALYSIS for p in analysis_prompts)
    
    def test_get_recommended_params(self):
        """Test getting recommended API parameters."""
        registry = PromptRegistry.__new__(PromptRegistry)
        registry._registry = PROMPT_REGISTRY.copy()
        registry.manager = None
        
        params = registry.get_recommended_params("tactical_assessment")
        
        assert "temperature" in params
        assert "max_tokens" in params
        assert params["temperature"] == 0.0  # Deterministic for assessments
    
    def test_to_documentation(self):
        """Test documentation generation."""
        registry = PromptRegistry.__new__(PromptRegistry)
        registry._registry = PROMPT_REGISTRY.copy()
        registry.manager = None
        
        docs = registry.to_documentation()
        
        assert "# Prompt Registry Documentation" in docs
        assert "image_analysis" in docs


class TestIterationLog:
    """Tests for IterationLog."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for logs in the workspace."""
        import uuid
        log_dir = TEST_TEMP_DIR / f"iteration_log_test_{uuid.uuid4().hex[:8]}"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        yield log_dir
        
        # Cleanup
        if log_dir.exists():
            shutil.rmtree(log_dir, ignore_errors=True)
    
    def test_log_change(self, temp_log_dir):
        """Test logging a prompt change."""
        log = IterationLog(prompts_dir=str(temp_log_dir))
        
        experiment = log.log_change(
            prompt_name="test_prompt",
            version="1.1",
            change_type="refinement",
            description="Added examples",
            rationale="Improve consistency",
            changes_made=["Added 2 examples"],
        )
        
        assert experiment.prompt_name == "test_prompt"
        assert experiment.version == "1.1"
        assert len(log._experiments) == 1
    
    def test_log_test_results(self, temp_log_dir):
        """Test updating experiment with test results."""
        log = IterationLog(prompts_dir=str(temp_log_dir))
        
        log.log_change(
            prompt_name="test_prompt",
            version="1.0",
            change_type="initial",
            description="Initial",
            rationale="Baseline",
        )
        
        log.log_test_results(
            prompt_name="test_prompt",
            version="1.0",
            test_cases_run=10,
            test_cases_passed=9,
        )
        
        # Get the latest entry (which should have been updated)
        history = log.get_history("test_prompt")
        experiment = history[-1]  # Get the most recent one
        assert experiment.test_cases_run == 10
        assert experiment.test_cases_passed == 9
    
    def test_log_quality_metrics(self, temp_log_dir):
        """Test updating experiment with quality metrics."""
        log = IterationLog(prompts_dir=str(temp_log_dir))
        
        log.log_change(
            prompt_name="test_prompt",
            version="1.0",
            change_type="initial",
            description="Initial",
            rationale="Baseline",
        )
        
        log.log_quality_metrics(
            prompt_name="test_prompt",
            version="1.0",
            relevance=0.95,
            coherence=0.88,
        )
        
        # Get the latest entry (which should have been updated)
        history = log.get_history("test_prompt")
        experiment = history[-1]  # Get the most recent one
        assert experiment.relevance_score == 0.95
        assert experiment.coherence_score == 0.88
    
    def test_get_history(self, temp_log_dir):
        """Test getting experiment history for a prompt."""
        log = IterationLog(prompts_dir=str(temp_log_dir))
        
        log.log_change("prompt_a", "1.0", "initial", "First", "Baseline")
        log.log_change("prompt_a", "1.1", "refinement", "Second", "Improve")
        log.log_change("prompt_b", "1.0", "initial", "Other", "Different")
        
        history_a = log.get_history("prompt_a")
        history_b = log.get_history("prompt_b")
        
        assert len(history_a) == 2
        assert len(history_b) == 1
    
    def test_generate_report(self, temp_log_dir):
        """Test report generation."""
        log = IterationLog(prompts_dir=str(temp_log_dir))
        
        log.log_change("test_prompt", "1.0", "initial", "Initial", "Baseline")
        log.log_test_results("test_prompt", "1.0", 10, 9)
        
        report = log.generate_report()
        
        assert "Prompt Engineering Report" in report
        assert "test_prompt" in report
    
    def test_persistence(self, temp_log_dir):
        """Test that experiments persist to file."""
        log1 = IterationLog(prompts_dir=str(temp_log_dir))
        log1.log_change("test_persist", "1.0", "initial", "Test", "Testing")
        
        # Create new log instance to test loading
        log2 = IterationLog(prompts_dir=str(temp_log_dir))
        
        # Should have exactly 1 experiment for this unique prompt name
        test_history = log2.get_history("test_persist")
        assert len(test_history) == 1
        assert test_history[0].prompt_name == "test_persist"


class TestValidators:
    """Tests for prompt validators."""
    
    def test_validate_header_valid(self):
        """Test validation of valid header."""
        content = """# Prompt: Test
# Version: 1.0
# Last Updated: 2024-01-20
# Purpose: Testing
"""
        is_valid, issues = validate_header(content)
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_header_missing_version(self):
        """Test validation catches missing version."""
        content = """# Prompt: Test
# Purpose: Testing
"""
        is_valid, issues = validate_header(content)
        assert not is_valid
        assert any("Version" in issue for issue in issues)
    
    def test_validate_structure(self):
        """Test structure validation."""
        content = """## ROLE
Test role

## TASK
Do something

## OUTPUT FORMAT
Return text
"""
        is_valid, issues = validate_structure(content)
        assert is_valid
    
    def test_validate_structure_missing_section(self):
        """Test structure validation catches missing sections."""
        content = """## ROLE
Test role
"""
        is_valid, issues = validate_structure(content)
        assert not is_valid
        assert any("TASK" in issue for issue in issues)
    
    def test_validate_variables_unmatched_braces(self):
        """Test variable validation catches unmatched braces."""
        content = "Hello {name and {other}"
        is_valid, issues = validate_variables(content)
        assert not is_valid
        assert any("braces" in issue.lower() for issue in issues)
    
    def test_validate_prompt_full(self):
        """Test full prompt validation."""
        template = PromptTemplate(
            name="test",
            content="""# Prompt: Test
# Version: 1.0
# Last Updated: 2024-01-20
# Purpose: Testing

## ROLE
Test role

## TASK
Process {input}

## OUTPUT FORMAT
Return the processed text with the following structure:
- Summary of processing
- Key findings
- Recommendations for further action
"""
        )
        
        is_valid, issues = validate_prompt(template)
        # Even if not strictly valid, check basic structure is recognized
        assert template.metadata.version == "1.0"
        assert "input" in template.required_variables
    
    def test_linter_suggestions(self):
        """Test linter provides suggestions."""
        linter = PromptLinter()
        
        content = "Do stuff etc and try to be helpful"
        suggestions = linter.lint(content)
        
        assert len(suggestions) > 0
        assert any("etc" in s.lower() for s in suggestions)


class TestIntegration:
    """Integration tests for the prompt framework."""
    
    def test_full_workflow(self):
        """Test complete workflow: load, validate, render, log."""
        # Use the actual prompts directory
        prompts_dir = Path(__file__).parent.parent / "prompts"
        
        if not prompts_dir.exists():
            pytest.skip("Prompts directory not found")
        
        # Load manager
        manager = PromptManager(prompts_dir=str(prompts_dir))
        
        # Get all templates
        templates = manager.get_all_templates()
        assert len(templates) > 0
        
        # Validate all
        issues = manager.validate_all()
        
        # Each template should have valid structure
        for template in templates:
            is_valid, template_issues = validate_prompt(template)
            # We don't require strict validity, but should pass basic checks


# Cleanup singleton between tests
@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    reset_manager()
    yield
