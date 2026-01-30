"""
Prompt Iteration Tracking - Tools for logging and analyzing prompt changes.

Provides utilities for tracking prompt experiments and maintaining an iteration log.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class PromptExperiment:
    """
    Record of a prompt experiment or iteration.
    
    Tracks changes to prompts along with their rationale and results.
    """
    prompt_name: str
    version: str
    timestamp: str
    change_type: str  # "initial", "refinement", "fix", "optimization", "rollback"
    description: str
    rationale: str
    changes_made: list[str] = field(default_factory=list)
    
    # Testing results
    test_cases_run: int = 0
    test_cases_passed: int = 0
    sample_outputs: list[str] = field(default_factory=list)
    
    # Quality metrics (optional)
    relevance_score: Optional[float] = None
    coherence_score: Optional[float] = None
    format_compliance: Optional[float] = None
    
    # A/B testing
    compared_with: Optional[str] = None  # Previous version compared against
    improvement_notes: Optional[str] = None
    
    def to_markdown(self) -> str:
        """Convert experiment to markdown format."""
        lines = [
            f"### v{self.version} ({self.timestamp})",
            f"**Type:** {self.change_type}",
            "",
            f"**Description:** {self.description}",
            "",
            f"**Rationale:** {self.rationale}",
            "",
        ]
        
        if self.changes_made:
            lines.append("**Changes:**")
            for change in self.changes_made:
                lines.append(f"- {change}")
            lines.append("")
        
        if self.test_cases_run > 0:
            lines.append(f"**Testing:** {self.test_cases_passed}/{self.test_cases_run} test cases passed")
            lines.append("")
        
        if self.relevance_score is not None:
            lines.append("**Quality Metrics:**")
            if self.relevance_score is not None:
                lines.append(f"- Relevance: {self.relevance_score:.2f}")
            if self.coherence_score is not None:
                lines.append(f"- Coherence: {self.coherence_score:.2f}")
            if self.format_compliance is not None:
                lines.append(f"- Format Compliance: {self.format_compliance:.2f}")
            lines.append("")
        
        if self.compared_with:
            lines.append(f"**Compared With:** v{self.compared_with}")
            if self.improvement_notes:
                lines.append(f"**Improvement Notes:** {self.improvement_notes}")
            lines.append("")
        
        return "\n".join(lines)


class IterationLog:
    """
    Manages the prompt iteration log.
    
    Tracks all changes to prompts for reproducibility and analysis.
    """
    
    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize the iteration log.
        
        Args:
            prompts_dir: Directory containing prompts and log files
        """
        self.prompts_dir = Path(prompts_dir)
        self.changelog_path = self.prompts_dir / "CHANGELOG.md"
        self.experiments_path = self.prompts_dir / "experiments.json"
        
        # In-memory experiment log
        self._experiments: list[PromptExperiment] = []
        self._load_experiments()
    
    def _load_experiments(self):
        """Load experiments from JSON file."""
        if self.experiments_path.exists():
            try:
                data = json.loads(self.experiments_path.read_text())
                self._experiments = [
                    PromptExperiment(**exp) for exp in data
                ]
            except (json.JSONDecodeError, TypeError):
                self._experiments = []
    
    def _save_experiments(self):
        """Save experiments to JSON file."""
        data = [asdict(exp) for exp in self._experiments]
        self.experiments_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8"
        )
    
    def add_experiment(self, experiment: PromptExperiment):
        """
        Add a new experiment to the log.
        
        Args:
            experiment: PromptExperiment to add
        """
        self._experiments.append(experiment)
        self._save_experiments()
        self._update_changelog()
    
    def log_change(
        self,
        prompt_name: str,
        version: str,
        change_type: str,
        description: str,
        rationale: str,
        changes_made: Optional[list[str]] = None,
    ) -> PromptExperiment:
        """
        Log a prompt change (convenience method).
        
        Args:
            prompt_name: Name of the prompt
            version: New version number
            change_type: Type of change
            description: What was changed
            rationale: Why the change was made
            changes_made: List of specific changes
            
        Returns:
            Created PromptExperiment
        """
        experiment = PromptExperiment(
            prompt_name=prompt_name,
            version=version,
            timestamp=datetime.now().strftime("%Y-%m-%d"),
            change_type=change_type,
            description=description,
            rationale=rationale,
            changes_made=changes_made or [],
        )
        self.add_experiment(experiment)
        return experiment
    
    def log_test_results(
        self,
        prompt_name: str,
        version: str,
        test_cases_run: int,
        test_cases_passed: int,
        sample_outputs: Optional[list[str]] = None,
    ):
        """
        Update an experiment with test results.
        
        Args:
            prompt_name: Name of the prompt
            version: Version that was tested
            test_cases_run: Number of test cases
            test_cases_passed: Number passed
            sample_outputs: Example outputs from testing
        """
        for exp in reversed(self._experiments):
            if exp.prompt_name == prompt_name and exp.version == version:
                exp.test_cases_run = test_cases_run
                exp.test_cases_passed = test_cases_passed
                exp.sample_outputs = sample_outputs or []
                self._save_experiments()
                self._update_changelog()
                return
    
    def log_quality_metrics(
        self,
        prompt_name: str,
        version: str,
        relevance: Optional[float] = None,
        coherence: Optional[float] = None,
        format_compliance: Optional[float] = None,
    ):
        """
        Update an experiment with quality metrics.
        
        Args:
            prompt_name: Name of the prompt
            version: Version that was evaluated
            relevance: Relevance score (0-1)
            coherence: Coherence score (0-1)
            format_compliance: Format compliance score (0-1)
        """
        for exp in reversed(self._experiments):
            if exp.prompt_name == prompt_name and exp.version == version:
                if relevance is not None:
                    exp.relevance_score = relevance
                if coherence is not None:
                    exp.coherence_score = coherence
                if format_compliance is not None:
                    exp.format_compliance = format_compliance
                self._save_experiments()
                self._update_changelog()
                return
    
    def get_history(self, prompt_name: str) -> list[PromptExperiment]:
        """
        Get the experiment history for a prompt.
        
        Args:
            prompt_name: Name of the prompt
            
        Returns:
            List of experiments in chronological order
        """
        return [
            exp for exp in self._experiments
            if exp.prompt_name == prompt_name
        ]
    
    def get_latest_version(self, prompt_name: str) -> Optional[str]:
        """Get the latest version number for a prompt."""
        history = self.get_history(prompt_name)
        if not history:
            return None
        return history[-1].version
    
    def _update_changelog(self):
        """Update the CHANGELOG.md file from experiments."""
        lines = [
            "# Prompt Iteration Log",
            "",
            "This file documents the evolution of prompt templates used in Sentinel RAG.",
            "",
            "## Version History",
            "",
        ]
        
        # Group experiments by prompt name
        by_prompt: dict[str, list[PromptExperiment]] = {}
        for exp in self._experiments:
            if exp.prompt_name not in by_prompt:
                by_prompt[exp.prompt_name] = []
            by_prompt[exp.prompt_name].append(exp)
        
        # Generate changelog sections
        for prompt_name in sorted(by_prompt.keys()):
            lines.append(f"## {prompt_name}.txt")
            lines.append("")
            
            # Reverse chronological order
            for exp in reversed(by_prompt[prompt_name]):
                lines.append(exp.to_markdown())
            
            lines.append("---")
            lines.append("")
        
        # Add planned improvements section
        lines.extend([
            "## Planned Improvements",
            "",
            "- [ ] Add few-shot examples for better consistency",
            "- [ ] Implement domain-specific terminology glossary",
            "- [ ] Test temperature variations for assessment diversity",
            "- [ ] Add chain-of-thought reasoning prompts",
            "- [ ] Create automated evaluation pipeline",
            "",
        ])
        
        self.changelog_path.write_text("\n".join(lines), encoding="utf-8")
    
    def generate_report(self) -> str:
        """Generate a summary report of all prompt experiments."""
        lines = [
            "# Prompt Engineering Report",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Summary",
            "",
            f"Total experiments logged: {len(self._experiments)}",
            "",
        ]
        
        # Calculate stats per prompt
        by_prompt: dict[str, list[PromptExperiment]] = {}
        for exp in self._experiments:
            if exp.prompt_name not in by_prompt:
                by_prompt[exp.prompt_name] = []
            by_prompt[exp.prompt_name].append(exp)
        
        lines.append("| Prompt | Versions | Latest | Tests Run | Pass Rate |")
        lines.append("|--------|----------|--------|-----------|-----------|")
        
        for prompt_name in sorted(by_prompt.keys()):
            exps = by_prompt[prompt_name]
            latest = exps[-1].version
            total_tests = sum(e.test_cases_run for e in exps)
            total_passed = sum(e.test_cases_passed for e in exps)
            pass_rate = f"{100 * total_passed / total_tests:.1f}%" if total_tests > 0 else "N/A"
            
            lines.append(f"| {prompt_name} | {len(exps)} | v{latest} | {total_tests} | {pass_rate} |")
        
        lines.append("")
        
        return "\n".join(lines)


def create_initial_log(prompts_dir: str = "prompts"):
    """
    Create initial iteration log entries for existing prompts.
    
    This bootstraps the iteration log from existing prompt files.
    """
    log = IterationLog(prompts_dir)
    prompts_path = Path(prompts_dir)
    
    # Read existing prompts and create initial entries
    for prompt_file in prompts_path.glob("*.txt"):
        prompt_name = prompt_file.stem
        
        # Skip if already has history
        if log.get_history(prompt_name):
            continue
        
        # Read version from file
        content = prompt_file.read_text()
        import re
        version_match = re.search(r"#\s*Version:\s*(\d+\.\d+)", content)
        version = version_match.group(1) if version_match else "1.0"
        
        log.log_change(
            prompt_name=prompt_name,
            version=version,
            change_type="initial",
            description=f"Initial version of {prompt_name} prompt",
            rationale="Baseline prompt for Sentinel RAG system",
            changes_made=["Created initial prompt structure"],
        )
    
    return log
