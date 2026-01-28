"""
Vision Processor - GPT-4V image analysis for sensor data.

Handles image encoding and analysis via OpenAI's vision API.
"""

import base64
import logging
import os
from typing import Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.timing import timed

logger = logging.getLogger(__name__)


class VisionProcessor:
    """
    Vision processor for analyzing sensor imagery with GPT-4V.
    
    Supports radar, sonar, and satellite imagery analysis.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-vision-preview",
        prompts_dir: str = "prompts",
        mock_mode: bool = False,
        max_tokens: int = 1024,
    ):
        """
        Initialize the vision processor.
        
        Args:
            api_key: OpenAI API key
            model: Vision model to use
            prompts_dir: Directory containing prompt templates
            mock_mode: If True, return mock responses
            max_tokens: Maximum tokens in response
        """
        self.client = AsyncOpenAI(api_key=api_key) if api_key else None
        self.model = model
        self.prompts_dir = prompts_dir
        self.mock_mode = mock_mode
        self.max_tokens = max_tokens
        
        # Load image analysis prompt
        self.analysis_prompt = self._load_prompt()
    
    def _load_prompt(self) -> str:
        """Load the image analysis prompt template."""
        prompt_path = os.path.join(self.prompts_dir, "image_analysis.txt")
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return self._default_prompt()
    
    def _default_prompt(self) -> str:
        """Default image analysis prompt."""
        return """You are a military intelligence analyst specializing in sensor data interpretation.

Analyze this sensor image and extract:
1. Type of sensor data (radar/sonar/satellite)
2. Objects or patterns detected (describe what you see)
3. Estimated quantities and positions
4. Any anomalies or notable features
5. Environmental conditions visible (if applicable)

Be specific about what you observe vs. what you infer.
Acknowledge uncertainty when present.
Use military terminology where appropriate.

Provide a structured analysis suitable for tactical assessment."""
    
    def _encode_image(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 for API."""
        return base64.standard_b64encode(image_bytes).decode("utf-8")
    
    def _detect_image_type(self, image_bytes: bytes) -> str:
        """Detect image MIME type from bytes."""
        # Check magic bytes
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        elif image_bytes[:2] == b'\xff\xd8':
            return "image/jpeg"
        elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
            return "image/gif"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            return "image/webp"
        else:
            return "image/png"  # Default assumption
    
    @timed("vision_analysis")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def analyze_image(
        self,
        image: bytes,
        additional_context: Optional[str] = None,
    ) -> str:
        """
        Analyze a sensor image using GPT-4V.
        
        Args:
            image: Image bytes
            additional_context: Optional additional context for analysis
            
        Returns:
            Textual analysis of the image
        """
        if self.mock_mode:
            return self._mock_analysis()
        
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        # Encode image
        base64_image = self._encode_image(image)
        image_type = self._detect_image_type(image)
        
        # Build prompt
        prompt = self.analysis_prompt
        if additional_context:
            prompt += f"\n\nAdditional Context:\n{additional_context}"
        
        # Make API call
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_type};base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=self.max_tokens,
        )
        
        result = response.choices[0].message.content
        logger.info(f"Image analysis completed: {len(result)} chars")
        return result
    
    def _mock_analysis(self) -> str:
        """Generate mock image analysis for demo/testing."""
        return """## Sensor Image Analysis

### Sensor Type
Radar composite display

### Objects Detected
1. **Primary Contacts**: 3-4 distinct radar returns in the central region
   - Spacing suggests organized formation
   - Return strength indicates medium-sized surface vessels
   
2. **Secondary Contacts**: 2 additional faint returns to the northeast
   - Possibly smaller vessels or electronic interference

### Pattern Analysis
- Formation appears to be a standard line-abreast configuration
- Heading estimated at approximately 270 degrees (westward)
- Speed estimated at 15-20 knots based on contact progression

### Notable Features
- Clear weather conditions (minimal clutter)
- No air contacts visible
- Consistent return strength suggests similar vessel types

### Uncertainties
- Exact vessel classification requires additional intelligence
- Formation could be commercial or military
- Cannot determine nationality from radar alone

### Assessment Confidence: Medium
This analysis is based on radar signatures only. Correlation with other intelligence sources recommended."""
    
    async def analyze_with_fallback(
        self,
        image: bytes,
        cache=None,
        additional_context: Optional[str] = None,
    ) -> dict:
        """
        Analyze image with caching and graceful degradation.
        
        Returns a result dict with success status and analysis or error.
        """
        try:
            # Check cache first
            if cache:
                import hashlib
                cache_key = hashlib.sha256(image[:1000]).hexdigest()
                cached = cache.get(cache_key)
                if cached:
                    return {
                        "success": True,
                        "analysis": cached,
                        "confidence": "high",
                        "note": "Using cached analysis"
                    }
            
            # Perform analysis
            analysis = await self.analyze_image(image, additional_context)
            
            # Cache result
            if cache:
                cache.set(cache_key, analysis, operation_type="vision")
            
            return {
                "success": True,
                "analysis": analysis,
                "confidence": "high"
            }
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            
            # Try to return cached response
            if cache:
                cached = cache.get(cache_key)
                if cached:
                    return {
                        "success": True,
                        "analysis": cached,
                        "confidence": "medium",
                        "note": f"Using cached response due to error: {str(e)}"
                    }
            
            return {
                "success": False,
                "analysis": None,
                "error": str(e)
            }
