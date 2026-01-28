"""
LLM Service - OpenAI GPT-4V integration for analysis and chat.

Handles tactical assessment generation and chat responses.
"""

import json
import logging
from typing import Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.cache import ResponseCache
from app.core.timing import timed
from app.models.documents import RetrievedDocument
from app.models.analysis import TacticalAssessment

logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM Service for generating tactical assessments and chat responses.
    
    Uses OpenAI GPT-4 with caching for reliability.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-vision-preview",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        cache: Optional[ResponseCache] = None,
        mock_mode: bool = False,
    ):
        """
        Initialize the LLM service.
        
        Args:
            api_key: OpenAI API key
            model: Model to use
            temperature: Sampling temperature (0 for deterministic)
            max_tokens: Maximum tokens in response
            cache: Optional response cache
            mock_mode: If True, return mock responses without API calls
        """
        self.client = AsyncOpenAI(api_key=api_key) if api_key else None
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache
        self.mock_mode = mock_mode
        
        # Load prompts
        self._load_prompts()
    
    def _load_prompts(self):
        """Load prompt templates from files."""
        import os
        prompts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")
        
        self.assessment_prompt = self._load_prompt(
            os.path.join(prompts_dir, "tactical_assessment.txt"),
            default=self._default_assessment_prompt()
        )
        self.chat_prompt = self._load_prompt(
            os.path.join(prompts_dir, "follow_up.txt"),
            default=self._default_chat_prompt()
        )
    
    def _load_prompt(self, path: str, default: str) -> str:
        """Load a prompt from file or return default."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return default
    
    def _default_assessment_prompt(self) -> str:
        """Default tactical assessment prompt."""
        return """You are a military intelligence analyst. Based on the image analysis and historical context, provide a tactical assessment.

IMAGE ANALYSIS:
{image_analysis}

HISTORICAL CONTEXT:
{context}

ADDITIONAL SITUATION CONTEXT:
{additional_context}

Provide your assessment in the following JSON format:
{{{{
    "sensor_type": "radar|sonar|satellite",
    "observations": ["observation 1", "observation 2"],
    "threat_level": "low|moderate|elevated|high|critical",
    "intent_analysis": "Analysis of likely enemy intent",
    "confidence": "low|medium|high",
    "recommended_actions": ["action 1", "action 2"],
    "caveats": ["caveat 1", "caveat 2"]
}}}}

Be specific about what you observe vs. what you infer. Acknowledge uncertainty when present."""
    
    def _default_chat_prompt(self) -> str:
        """Default chat follow-up prompt."""
        return """You are a military intelligence analyst assistant. Answer follow-up questions about the tactical assessment.

PREVIOUS CONVERSATION:
{conversation}

PREVIOUS ANALYSIS:
{previous_analysis}

RELEVANT CONTEXT:
{context}

USER QUESTION:
{question}

Provide a helpful, specific answer based on the available information. Be clear about uncertainty."""
    
    @timed("llm_assessment")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate_assessment(
        self,
        image_analysis: str,
        retrieved_context: list[RetrievedDocument],
        additional_context: Optional[str] = None,
    ) -> TacticalAssessment | str:
        """
        Generate a tactical assessment based on image analysis and context.
        
        Args:
            image_analysis: Analysis from vision model
            retrieved_context: Retrieved documents from RAG
            additional_context: Additional user-provided context
            
        Returns:
            Structured tactical assessment
        """
        # Build context from retrieved documents
        context_text = "\n\n".join([
            f"[{doc.metadata.get('title', 'Unknown')}]\n{doc.content}"
            for doc in retrieved_context
        ]) or "No historical context available."
        
        # Format prompt
        prompt = self.assessment_prompt.format(
            image_analysis=image_analysis,
            context=context_text,
            additional_context=additional_context or "None provided.",
        )
        
        # Check cache
        if self.cache:
            cache_key = self.cache._generate_key(prompt)
            cached = self.cache.get(cache_key)
            if cached:
                try:
                    return TacticalAssessment(**json.loads(cached))
                except (json.JSONDecodeError, ValueError):
                    return cached
        
        # Mock mode
        if self.mock_mode:
            return self._mock_assessment(image_analysis)
        
        # Make API call
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        response = await self.client.chat.completions.create(
            model=self.model.replace("-vision-preview", "-turbo"),  # Use text model for synthesis
            messages=[
                {"role": "system", "content": "You are a military intelligence analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        result = response.choices[0].message.content
        
        # Cache result
        if self.cache:
            self.cache.set(cache_key, result, operation_type="assessment")
        
        # Try to parse as JSON
        try:
            # Extract JSON from response
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                return TacticalAssessment(**json.loads(json_str))
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Could not parse assessment as JSON: {e}")
        
        return result
    
    @timed("llm_chat")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate_chat_response(
        self,
        conversation: str,
        question: str,
        retrieved_context: str,
        previous_analysis: Optional[str] = None,
    ) -> str:
        """
        Generate a chat response for follow-up questions.
        
        Args:
            conversation: Previous conversation history
            question: Current user question
            retrieved_context: Context from RAG retrieval
            previous_analysis: Previous tactical assessment
            
        Returns:
            Assistant response text
        """
        prompt = self.chat_prompt.format(
            conversation=conversation,
            previous_analysis=previous_analysis or "No previous analysis.",
            context=retrieved_context or "No additional context available.",
            question=question,
        )
        
        # Check cache
        if self.cache:
            cache_key = self.cache._generate_key(prompt)
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        # Mock mode
        if self.mock_mode:
            return self._mock_chat_response(question)
        
        # Make API call
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        response = await self.client.chat.completions.create(
            model=self.model.replace("-vision-preview", "-turbo"),
            messages=[
                {"role": "system", "content": "You are a military intelligence analyst assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        result = response.choices[0].message.content
        
        # Cache result
        if self.cache:
            self.cache.set(cache_key, result, operation_type="chat")
        
        return result
    
    def _mock_assessment(self, image_analysis: str) -> TacticalAssessment:
        """Generate mock assessment for demo/testing."""
        return TacticalAssessment(
            sensor_type="radar",
            observations=[
                "Multiple surface contacts detected in formation",
                "Estimated 3-5 vessels based on radar returns",
                "Movement pattern suggests coordinated activity"
            ],
            threat_level="moderate",
            intent_analysis="The formation pattern is consistent with a standard convoy or patrol formation. Based on historical patterns, this could indicate routine naval operations or a reconnaissance mission. Further monitoring is recommended to confirm intent.",
            confidence="medium",
            recommended_actions=[
                "Continue monitoring contact positions",
                "Cross-reference with known shipping lanes",
                "Prepare for potential escalation scenarios"
            ],
            caveats=[
                "Analysis based on simulated data",
                "Real-world assessment would require additional intelligence sources",
                "This is a demonstration system"
            ]
        )
    
    def _mock_chat_response(self, question: str) -> str:
        """Generate mock chat response for demo/testing."""
        return f"""Based on the available intelligence and historical context, I can provide the following response to your question:

{question}

The current analysis indicates standard operational patterns. Historical records suggest similar activities have been observed in this region previously. However, I recommend continued monitoring and correlation with additional intelligence sources for a more comprehensive assessment.

Note: This is a demonstration response from the mock system."""
