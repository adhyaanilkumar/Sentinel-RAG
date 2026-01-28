"""
Unit tests for Vision Processor.

Tests image analysis and encoding.
"""

import pytest
from app.core.vision import VisionProcessor


class TestVisionProcessor:
    """Tests for VisionProcessor class."""
    
    @pytest.fixture
    def mock_vision(self):
        """Create vision processor in mock mode."""
        return VisionProcessor(
            api_key="test-key",
            model="gpt-4-vision-preview",
            mock_mode=True,
        )
    
    def test_image_type_detection_png(self, mock_vision, sample_image_bytes):
        """Test PNG image type detection."""
        image_type = mock_vision._detect_image_type(sample_image_bytes)
        assert image_type == "image/png"
    
    def test_image_type_detection_jpeg(self, mock_vision):
        """Test JPEG image type detection."""
        # JPEG magic bytes
        jpeg_bytes = b'\xff\xd8\xff\xe0' + b'\x00' * 100
        image_type = mock_vision._detect_image_type(jpeg_bytes)
        assert image_type == "image/jpeg"
    
    def test_image_type_detection_unknown(self, mock_vision):
        """Test default image type for unknown format."""
        unknown_bytes = b'\x00\x00\x00\x00'
        image_type = mock_vision._detect_image_type(unknown_bytes)
        assert image_type == "image/png"  # Default
    
    def test_encode_image(self, mock_vision, sample_image_bytes):
        """Test image encoding to base64."""
        encoded = mock_vision._encode_image(sample_image_bytes)
        
        assert isinstance(encoded, str)
        # Verify it's valid base64
        import base64
        decoded = base64.b64decode(encoded)
        assert decoded == sample_image_bytes
    
    @pytest.mark.asyncio
    async def test_mock_analysis(self, mock_vision, sample_image_bytes):
        """Test mock image analysis."""
        analysis = await mock_vision.analyze_image(sample_image_bytes)
        
        assert isinstance(analysis, str)
        assert len(analysis) > 0
        assert "Sensor" in analysis or "radar" in analysis.lower()
    
    @pytest.mark.asyncio
    async def test_analyze_with_fallback_mock(self, mock_vision, sample_image_bytes):
        """Test analysis with fallback in mock mode."""
        result = await mock_vision.analyze_with_fallback(sample_image_bytes)
        
        assert result["success"] == True
        assert "analysis" in result
        assert result["analysis"] is not None
    
    def test_default_prompt_loaded(self, mock_vision):
        """Test that default prompt is available."""
        assert mock_vision.analysis_prompt is not None
        assert len(mock_vision.analysis_prompt) > 100
        assert "military" in mock_vision.analysis_prompt.lower() or "sensor" in mock_vision.analysis_prompt.lower()


class TestVisionProcessorIntegration:
    """Integration-style tests (still using mock mode)."""
    
    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self, sample_image_bytes):
        """Test complete analysis pipeline."""
        processor = VisionProcessor(
            api_key="test-key",
            mock_mode=True,
        )
        
        result = await processor.analyze_with_fallback(sample_image_bytes)
        
        assert result["success"] == True
        assert "analysis" in result
        
        # Analysis should have structured content
        analysis = result["analysis"]
        assert isinstance(analysis, str)
