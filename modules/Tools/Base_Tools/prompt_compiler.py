"""Prompt Compiler for Image Generation.

Optimizes and transforms prompts for different image generation providers.
Different models have different prompt requirements:
- FLUX: Natural language, detailed descriptions
- SDXL: Quality tags, weight syntax (word:1.5)
- Midjourney-style: Short, punchy, artistic terms
- Ideogram: Text rendering focus
- Imagen: Simple, direct descriptions
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import re


class PromptStyle(Enum):
    """Target prompt style for compilation."""
    NATURAL = "natural"           # FLUX, Gemini - conversational descriptions
    TAGGED = "tagged"             # SDXL, SD1.5 - quality tags, weights
    ARTISTIC = "artistic"         # Midjourney-style - short, evocative
    DIRECT = "direct"             # Imagen - simple, clear descriptions
    TEXT_FOCUSED = "text_focused" # Ideogram - text rendering emphasis


@dataclass
class PromptComponents:
    """Parsed prompt components for transformation."""
    subject: str = ""
    style: str = ""
    lighting: str = ""
    camera: str = ""
    quality_tags: list[str] = field(default_factory=list)
    negative_aspects: list[str] = field(default_factory=list)
    text_content: str = ""  # For text-in-image requests
    raw: str = ""


class PromptCompiler:
    """Compiles and optimizes prompts for different image generation models."""
    
    # Model family to style mapping
    MODEL_STYLES: dict[str, PromptStyle] = {
        # FLUX family - natural language
        "flux": PromptStyle.NATURAL,
        "flux-dev": PromptStyle.NATURAL,
        "flux-schnell": PromptStyle.NATURAL,
        "flux-pro": PromptStyle.NATURAL,
        "flux-1.1-pro": PromptStyle.NATURAL,
        "flux-realism": PromptStyle.NATURAL,
        
        # SDXL family - tagged style
        "sdxl": PromptStyle.TAGGED,
        "stable-diffusion-xl": PromptStyle.TAGGED,
        "sd3": PromptStyle.TAGGED,
        "sd3.5": PromptStyle.TAGGED,
        "stable-diffusion": PromptStyle.TAGGED,
        "juggernaut": PromptStyle.TAGGED,
        "dreamshaper": PromptStyle.TAGGED,
        
        # Artistic style
        "midjourney": PromptStyle.ARTISTIC,
        "niji": PromptStyle.ARTISTIC,
        
        # Direct/simple
        "imagen": PromptStyle.DIRECT,
        "gemini": PromptStyle.DIRECT,
        "dall-e": PromptStyle.DIRECT,
        "grok": PromptStyle.DIRECT,
        "aurora": PromptStyle.DIRECT,
        
        # Text-focused
        "ideogram": PromptStyle.TEXT_FOCUSED,
        "recraft": PromptStyle.TEXT_FOCUSED,
    }
    
    # Quality enhancers by style
    QUALITY_ENHANCERS: dict[PromptStyle, list[str]] = {
        PromptStyle.NATURAL: [
            "highly detailed",
            "professional quality",
            "sharp focus",
        ],
        PromptStyle.TAGGED: [
            "masterpiece",
            "best quality", 
            "highly detailed",
            "8k",
            "sharp focus",
            "professional",
        ],
        PromptStyle.ARTISTIC: [
            "stunning",
            "award winning",
            "trending",
        ],
        PromptStyle.DIRECT: [],  # Keep simple
        PromptStyle.TEXT_FOCUSED: [
            "clear text",
            "legible",
            "professional typography",
        ],
    }
    
    # Style keywords to detect
    STYLE_KEYWORDS = [
        "anime", "realistic", "photorealistic", "cartoon", "oil painting",
        "watercolor", "sketch", "digital art", "3d render", "cinematic",
        "fantasy", "sci-fi", "vintage", "retro", "minimalist", "abstract",
        "impressionist", "surreal", "gothic", "steampunk", "cyberpunk",
    ]
    
    # Lighting keywords
    LIGHTING_KEYWORDS = [
        "dramatic lighting", "soft lighting", "golden hour", "blue hour",
        "studio lighting", "natural light", "backlit", "rim lighting",
        "neon lighting", "candlelight", "moonlight", "sunlight",
        "volumetric lighting", "ambient occlusion", "global illumination",
    ]
    
    # Camera/composition keywords
    CAMERA_KEYWORDS = [
        "close-up", "wide shot", "portrait", "landscape", "macro",
        "aerial view", "bird's eye view", "worm's eye view", "dutch angle",
        "bokeh", "shallow depth of field", "long exposure", "tilt shift",
        "fisheye", "panoramic", "symmetrical", "rule of thirds",
    ]
    
    def __init__(self):
        """Initialize the prompt compiler."""
        pass
    
    def get_style_for_model(self, model: str) -> PromptStyle:
        """Determine the appropriate prompt style for a model.
        
        Args:
            model: Model name or identifier
            
        Returns:
            PromptStyle appropriate for the model
        """
        model_lower = model.lower()
        
        # Check for exact or partial matches
        for key, style in self.MODEL_STYLES.items():
            if key in model_lower:
                return style
        
        # Default to natural for unknown models
        return PromptStyle.NATURAL
    
    def parse_prompt(self, prompt: str) -> PromptComponents:
        """Parse a prompt into its component parts.
        
        Args:
            prompt: Raw user prompt
            
        Returns:
            PromptComponents with parsed elements
        """
        components = PromptComponents(raw=prompt)
        prompt_lower = prompt.lower()
        
        # Extract style
        for keyword in self.STYLE_KEYWORDS:
            if keyword in prompt_lower:
                components.style = keyword
                break
        
        # Extract lighting
        for keyword in self.LIGHTING_KEYWORDS:
            if keyword in prompt_lower:
                components.lighting = keyword
                break
        
        # Extract camera/composition
        for keyword in self.CAMERA_KEYWORDS:
            if keyword in prompt_lower:
                components.camera = keyword
                break
        
        # Extract text content (for text-in-image)
        text_match = re.search(r'"([^"]+)"', prompt)
        if text_match:
            components.text_content = text_match.group(1)
        
        # The subject is the core content (simplified extraction)
        # Remove detected keywords to isolate subject
        subject = prompt
        for keyword in [components.style, components.lighting, components.camera]:
            if keyword:
                subject = re.sub(re.escape(keyword), "", subject, flags=re.IGNORECASE)
        
        # Clean up subject
        subject = re.sub(r'\s+', ' ', subject).strip()
        components.subject = subject
        
        return components
    
    def compile(
        self,
        prompt: str,
        model: str,
        *,
        enhance_quality: bool = True,
        target_style: PromptStyle | None = None,
    ) -> str:
        """Compile a prompt optimized for a specific model.
        
        Args:
            prompt: User's raw prompt
            model: Target model name
            enhance_quality: Whether to add quality enhancers
            target_style: Override auto-detected style
            
        Returns:
            Optimized prompt for the model
        """
        style = target_style or self.get_style_for_model(model)
        components = self.parse_prompt(prompt)
        
        if style == PromptStyle.NATURAL:
            return self._compile_natural(components, enhance_quality)
        elif style == PromptStyle.TAGGED:
            return self._compile_tagged(components, enhance_quality)
        elif style == PromptStyle.ARTISTIC:
            return self._compile_artistic(components, enhance_quality)
        elif style == PromptStyle.DIRECT:
            return self._compile_direct(components)
        elif style == PromptStyle.TEXT_FOCUSED:
            return self._compile_text_focused(components, enhance_quality)
        
        return prompt  # Fallback to original
    
    def _compile_natural(
        self,
        components: PromptComponents,
        enhance_quality: bool,
    ) -> str:
        """Compile for natural language models (FLUX, etc.)."""
        parts = [components.raw]
        
        if enhance_quality:
            # Add natural quality descriptors
            enhancers = self.QUALITY_ENHANCERS[PromptStyle.NATURAL]
            # Only add if not already present
            for enhancer in enhancers:
                if enhancer.lower() not in components.raw.lower():
                    parts.append(enhancer)
                    break  # Just one enhancer for natural style
        
        return ", ".join(parts)
    
    def _compile_tagged(
        self,
        components: PromptComponents,
        enhance_quality: bool,
    ) -> str:
        """Compile for tagged style models (SDXL, SD3, etc.)."""
        tags = []
        
        if enhance_quality:
            # Quality tags first
            tags.extend(self.QUALITY_ENHANCERS[PromptStyle.TAGGED][:3])
        
        # Main subject
        tags.append(components.raw)
        
        # Add detected style emphasis
        if components.style and components.style not in components.raw.lower():
            tags.append(components.style)
        
        return ", ".join(tags)
    
    def _compile_artistic(
        self,
        components: PromptComponents,
        enhance_quality: bool,
    ) -> str:
        """Compile for artistic/Midjourney style."""
        parts = []
        
        # Subject first, concise
        subject = components.subject or components.raw
        # Shorten if too long
        if len(subject) > 100:
            words = subject.split()
            subject = " ".join(words[:15])
        parts.append(subject)
        
        # Style
        if components.style:
            parts.append(components.style)
        
        # Artistic quality terms
        if enhance_quality:
            parts.extend(self.QUALITY_ENHANCERS[PromptStyle.ARTISTIC])
        
        # Aspect ratio hint style (common in artistic prompts)
        prompt = ", ".join(parts)
        return f"{prompt} --v 6"  # Midjourney-style version tag
    
    def _compile_direct(self, components: PromptComponents) -> str:
        """Compile for direct style (Imagen, DALL-E)."""
        # Keep it simple and clear
        # Remove excessive modifiers, keep core subject
        return components.raw.strip()
    
    def _compile_text_focused(
        self,
        components: PromptComponents,
        enhance_quality: bool,
    ) -> str:
        """Compile for text-rendering focused models (Ideogram, Recraft)."""
        parts = []
        
        # Emphasize text content if present
        if components.text_content:
            parts.append(f'Text "{components.text_content}" prominently displayed')
            # Add rest of prompt
            without_text = re.sub(r'"[^"]+"', '', components.raw)
            parts.append(without_text.strip())
        else:
            parts.append(components.raw)
        
        if enhance_quality:
            parts.extend(self.QUALITY_ENHANCERS[PromptStyle.TEXT_FOCUSED])
        
        return ", ".join(filter(None, parts))
    
    def compile_negative(
        self,
        model: str,
        custom_negatives: list[str] | None = None,
    ) -> str:
        """Generate a negative prompt for the model.
        
        Args:
            model: Target model name
            custom_negatives: Additional negative terms
            
        Returns:
            Negative prompt string
        """
        style = self.get_style_for_model(model)
        
        # Common negatives
        negatives = [
            "blurry",
            "low quality",
            "distorted",
            "deformed",
        ]
        
        # Style-specific negatives
        if style == PromptStyle.TAGGED:
            negatives.extend([
                "worst quality",
                "jpeg artifacts",
                "signature",
                "watermark",
                "username",
                "bad anatomy",
                "bad hands",
                "extra fingers",
            ])
        elif style == PromptStyle.ARTISTIC:
            negatives.extend([
                "ugly",
                "amateur",
                "poorly drawn",
            ])
        elif style == PromptStyle.TEXT_FOCUSED:
            negatives.extend([
                "misspelled",
                "illegible text",
                "garbled text",
            ])
        
        if custom_negatives:
            negatives.extend(custom_negatives)
        
        return ", ".join(negatives)
    
    def suggest_enhancements(self, prompt: str) -> dict[str, Any]:
        """Suggest enhancements for a prompt.
        
        Args:
            prompt: User's prompt
            
        Returns:
            Dictionary with suggestions
        """
        components = self.parse_prompt(prompt)
        suggestions: dict[str, Any] = {
            "has_style": bool(components.style),
            "has_lighting": bool(components.lighting),
            "has_camera": bool(components.camera),
            "has_text": bool(components.text_content),
            "recommendations": [],
        }
        
        if not components.style:
            suggestions["recommendations"].append(
                "Consider adding a style (e.g., 'photorealistic', 'oil painting', 'anime')"
            )
        
        if not components.lighting:
            suggestions["recommendations"].append(
                "Consider specifying lighting (e.g., 'dramatic lighting', 'golden hour')"
            )
        
        if not components.camera and len(prompt) > 50:
            suggestions["recommendations"].append(
                "Consider adding composition details (e.g., 'close-up', 'wide shot')"
            )
        
        return suggestions


# Module-level instance for convenience
_compiler: PromptCompiler | None = None


def get_compiler() -> PromptCompiler:
    """Get the singleton prompt compiler instance."""
    global _compiler
    if _compiler is None:
        _compiler = PromptCompiler()
    return _compiler


def compile_prompt(
    prompt: str,
    model: str,
    *,
    enhance_quality: bool = True,
) -> str:
    """Convenience function to compile a prompt.
    
    Args:
        prompt: User's raw prompt
        model: Target model name
        enhance_quality: Whether to add quality enhancers
        
    Returns:
        Optimized prompt
    """
    return get_compiler().compile(prompt, model, enhance_quality=enhance_quality)


def compile_negative_prompt(
    model: str,
    custom_negatives: list[str] | None = None,
) -> str:
    """Convenience function to generate negative prompt.
    
    Args:
        model: Target model name
        custom_negatives: Additional negative terms
        
    Returns:
        Negative prompt string
    """
    return get_compiler().compile_negative(model, custom_negatives)
