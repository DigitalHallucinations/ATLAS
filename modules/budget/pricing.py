"""Pricing registry for LLM and media generation providers.

Maintains current pricing data for all supported providers and models,
with methods for cost calculation and price updates.

Pricing data is based on provider documentation and updated periodically.
All prices are in USD per unit (per 1K tokens, per image, etc.).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from modules.logging.logger import setup_logger

if TYPE_CHECKING:
    from ATLAS.config import ConfigManager

logger = setup_logger(__name__)


class PricingUnit(Enum):
    """Unit of measurement for pricing."""

    PER_1K_TOKENS = "per_1k_tokens"
    PER_1M_TOKENS = "per_1m_tokens"
    PER_IMAGE = "per_image"
    PER_SECOND = "per_second"
    PER_MINUTE = "per_minute"
    PER_CHARACTER = "per_character"
    PER_1K_CHARACTERS = "per_1k_characters"
    PER_REQUEST = "per_request"


def _quantize(value: Decimal, places: int = 8) -> Decimal:
    """Quantize decimal for pricing precision."""
    return value.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)


def _to_decimal(value: Any) -> Decimal:
    """Convert value to Decimal."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


@dataclass
class ModelPricing:
    """Pricing information for a specific model.

    Attributes:
        model: Model identifier.
        provider: Provider name.
        input_price: Price per input unit.
        output_price: Price per output unit (for LLMs).
        unit: Pricing unit.
        image_prices: Size-based pricing for image models.
        cached_input_price: Discounted price for cached inputs.
        batch_input_price: Discounted price for batch API.
        batch_output_price: Discounted price for batch API outputs.
        effective_date: When this pricing became effective.
        notes: Additional pricing notes.
    """

    model: str
    provider: str
    input_price: Decimal
    output_price: Optional[Decimal] = None
    unit: PricingUnit = PricingUnit.PER_1M_TOKENS
    image_prices: Dict[str, Decimal] = field(default_factory=dict)
    cached_input_price: Optional[Decimal] = None
    batch_input_price: Optional[Decimal] = None
    batch_output_price: Optional[Decimal] = None
    effective_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        """Normalize pricing values."""
        self.input_price = _to_decimal(self.input_price)
        if self.output_price is not None:
            self.output_price = _to_decimal(self.output_price)
        if self.cached_input_price is not None:
            self.cached_input_price = _to_decimal(self.cached_input_price)
        if self.batch_input_price is not None:
            self.batch_input_price = _to_decimal(self.batch_input_price)
        if self.batch_output_price is not None:
            self.batch_output_price = _to_decimal(self.batch_output_price)

        normalized_images = {}
        for size, price in self.image_prices.items():
            normalized_images[size] = _to_decimal(price)
        self.image_prices = normalized_images


# =============================================================================
# LLM Provider Pricing (as of January 2026)
# Prices in USD per 1M tokens unless otherwise noted
# =============================================================================

OPENAI_PRICING: Dict[str, ModelPricing] = {
    # GPT-4o family
    "gpt-4o": ModelPricing(
        model="gpt-4o",
        provider="OpenAI",
        input_price=Decimal("2.50"),
        output_price=Decimal("10.00"),
        cached_input_price=Decimal("1.25"),
    ),
    "gpt-4o-2024-11-20": ModelPricing(
        model="gpt-4o-2024-11-20",
        provider="OpenAI",
        input_price=Decimal("2.50"),
        output_price=Decimal("10.00"),
        cached_input_price=Decimal("1.25"),
    ),
    "gpt-4o-2024-08-06": ModelPricing(
        model="gpt-4o-2024-08-06",
        provider="OpenAI",
        input_price=Decimal("2.50"),
        output_price=Decimal("10.00"),
        cached_input_price=Decimal("1.25"),
    ),
    "gpt-4o-mini": ModelPricing(
        model="gpt-4o-mini",
        provider="OpenAI",
        input_price=Decimal("0.15"),
        output_price=Decimal("0.60"),
        cached_input_price=Decimal("0.075"),
    ),
    "gpt-4o-mini-2024-07-18": ModelPricing(
        model="gpt-4o-mini-2024-07-18",
        provider="OpenAI",
        input_price=Decimal("0.15"),
        output_price=Decimal("0.60"),
        cached_input_price=Decimal("0.075"),
    ),
    # GPT-4 Turbo
    "gpt-4-turbo": ModelPricing(
        model="gpt-4-turbo",
        provider="OpenAI",
        input_price=Decimal("10.00"),
        output_price=Decimal("30.00"),
    ),
    "gpt-4-turbo-2024-04-09": ModelPricing(
        model="gpt-4-turbo-2024-04-09",
        provider="OpenAI",
        input_price=Decimal("10.00"),
        output_price=Decimal("30.00"),
    ),
    # GPT-4
    "gpt-4": ModelPricing(
        model="gpt-4",
        provider="OpenAI",
        input_price=Decimal("30.00"),
        output_price=Decimal("60.00"),
    ),
    "gpt-4-32k": ModelPricing(
        model="gpt-4-32k",
        provider="OpenAI",
        input_price=Decimal("60.00"),
        output_price=Decimal("120.00"),
    ),
    # GPT-3.5 Turbo
    "gpt-3.5-turbo": ModelPricing(
        model="gpt-3.5-turbo",
        provider="OpenAI",
        input_price=Decimal("0.50"),
        output_price=Decimal("1.50"),
    ),
    "gpt-3.5-turbo-0125": ModelPricing(
        model="gpt-3.5-turbo-0125",
        provider="OpenAI",
        input_price=Decimal("0.50"),
        output_price=Decimal("1.50"),
    ),
    # o1 reasoning models
    "o1": ModelPricing(
        model="o1",
        provider="OpenAI",
        input_price=Decimal("15.00"),
        output_price=Decimal("60.00"),
        cached_input_price=Decimal("7.50"),
    ),
    "o1-2024-12-17": ModelPricing(
        model="o1-2024-12-17",
        provider="OpenAI",
        input_price=Decimal("15.00"),
        output_price=Decimal("60.00"),
        cached_input_price=Decimal("7.50"),
    ),
    "o1-preview": ModelPricing(
        model="o1-preview",
        provider="OpenAI",
        input_price=Decimal("15.00"),
        output_price=Decimal("60.00"),
        cached_input_price=Decimal("7.50"),
    ),
    "o1-mini": ModelPricing(
        model="o1-mini",
        provider="OpenAI",
        input_price=Decimal("3.00"),
        output_price=Decimal("12.00"),
        cached_input_price=Decimal("1.50"),
    ),
    "o3-mini": ModelPricing(
        model="o3-mini",
        provider="OpenAI",
        input_price=Decimal("1.10"),
        output_price=Decimal("4.40"),
        cached_input_price=Decimal("0.55"),
    ),
    # Embeddings
    "text-embedding-3-small": ModelPricing(
        model="text-embedding-3-small",
        provider="OpenAI",
        input_price=Decimal("0.02"),
        output_price=None,
    ),
    "text-embedding-3-large": ModelPricing(
        model="text-embedding-3-large",
        provider="OpenAI",
        input_price=Decimal("0.13"),
        output_price=None,
    ),
    "text-embedding-ada-002": ModelPricing(
        model="text-embedding-ada-002",
        provider="OpenAI",
        input_price=Decimal("0.10"),
        output_price=None,
    ),
}

ANTHROPIC_PRICING: Dict[str, ModelPricing] = {
    # Claude 4 (Opus)
    "claude-sonnet-4-20250514": ModelPricing(
        model="claude-sonnet-4-20250514",
        provider="Anthropic",
        input_price=Decimal("3.00"),
        output_price=Decimal("15.00"),
        cached_input_price=Decimal("0.30"),
    ),
    # Claude 3.5 family
    "claude-3-5-sonnet-20241022": ModelPricing(
        model="claude-3-5-sonnet-20241022",
        provider="Anthropic",
        input_price=Decimal("3.00"),
        output_price=Decimal("15.00"),
        cached_input_price=Decimal("0.30"),
    ),
    "claude-3-5-sonnet-20240620": ModelPricing(
        model="claude-3-5-sonnet-20240620",
        provider="Anthropic",
        input_price=Decimal("3.00"),
        output_price=Decimal("15.00"),
        cached_input_price=Decimal("0.30"),
    ),
    "claude-3-5-haiku-20241022": ModelPricing(
        model="claude-3-5-haiku-20241022",
        provider="Anthropic",
        input_price=Decimal("0.80"),
        output_price=Decimal("4.00"),
        cached_input_price=Decimal("0.08"),
    ),
    # Claude 3 family
    "claude-3-opus-20240229": ModelPricing(
        model="claude-3-opus-20240229",
        provider="Anthropic",
        input_price=Decimal("15.00"),
        output_price=Decimal("75.00"),
        cached_input_price=Decimal("1.50"),
    ),
    "claude-3-sonnet-20240229": ModelPricing(
        model="claude-3-sonnet-20240229",
        provider="Anthropic",
        input_price=Decimal("3.00"),
        output_price=Decimal("15.00"),
        cached_input_price=Decimal("0.30"),
    ),
    "claude-3-haiku-20240307": ModelPricing(
        model="claude-3-haiku-20240307",
        provider="Anthropic",
        input_price=Decimal("0.25"),
        output_price=Decimal("1.25"),
        cached_input_price=Decimal("0.03"),
    ),
}

GOOGLE_PRICING: Dict[str, ModelPricing] = {
    # Gemini 2.0
    "gemini-2.0-flash": ModelPricing(
        model="gemini-2.0-flash",
        provider="Google",
        input_price=Decimal("0.10"),
        output_price=Decimal("0.40"),
        notes="Free tier: 1500 RPD, 1M TPM",
    ),
    "gemini-2.0-flash-lite": ModelPricing(
        model="gemini-2.0-flash-lite",
        provider="Google",
        input_price=Decimal("0.075"),
        output_price=Decimal("0.30"),
    ),
    # Gemini 1.5
    "gemini-1.5-pro": ModelPricing(
        model="gemini-1.5-pro",
        provider="Google",
        input_price=Decimal("1.25"),
        output_price=Decimal("5.00"),
        notes="<=128K context; 2.50/10.00 for >128K",
    ),
    "gemini-1.5-flash": ModelPricing(
        model="gemini-1.5-flash",
        provider="Google",
        input_price=Decimal("0.075"),
        output_price=Decimal("0.30"),
        notes="<=128K context; 0.15/0.60 for >128K",
    ),
    "gemini-1.5-flash-8b": ModelPricing(
        model="gemini-1.5-flash-8b",
        provider="Google",
        input_price=Decimal("0.0375"),
        output_price=Decimal("0.15"),
    ),
    # Gemini 1.0
    "gemini-1.0-pro": ModelPricing(
        model="gemini-1.0-pro",
        provider="Google",
        input_price=Decimal("0.50"),
        output_price=Decimal("1.50"),
    ),
    # Embeddings
    "text-embedding-004": ModelPricing(
        model="text-embedding-004",
        provider="Google",
        input_price=Decimal("0.00"),
        output_price=None,
        notes="Free for now",
    ),
}

MISTRAL_PRICING: Dict[str, ModelPricing] = {
    # Premier models
    "mistral-large-latest": ModelPricing(
        model="mistral-large-latest",
        provider="Mistral",
        input_price=Decimal("2.00"),
        output_price=Decimal("6.00"),
    ),
    "mistral-large-2411": ModelPricing(
        model="mistral-large-2411",
        provider="Mistral",
        input_price=Decimal("2.00"),
        output_price=Decimal("6.00"),
    ),
    "pixtral-large-latest": ModelPricing(
        model="pixtral-large-latest",
        provider="Mistral",
        input_price=Decimal("2.00"),
        output_price=Decimal("6.00"),
    ),
    # Medium models
    "mistral-medium-latest": ModelPricing(
        model="mistral-medium-latest",
        provider="Mistral",
        input_price=Decimal("2.70"),
        output_price=Decimal("8.10"),
    ),
    "mistral-small-latest": ModelPricing(
        model="mistral-small-latest",
        provider="Mistral",
        input_price=Decimal("0.20"),
        output_price=Decimal("0.60"),
    ),
    # Codestral
    "codestral-latest": ModelPricing(
        model="codestral-latest",
        provider="Mistral",
        input_price=Decimal("0.30"),
        output_price=Decimal("0.90"),
    ),
    # Ministral
    "ministral-8b-latest": ModelPricing(
        model="ministral-8b-latest",
        provider="Mistral",
        input_price=Decimal("0.10"),
        output_price=Decimal("0.10"),
    ),
    "ministral-3b-latest": ModelPricing(
        model="ministral-3b-latest",
        provider="Mistral",
        input_price=Decimal("0.04"),
        output_price=Decimal("0.04"),
    ),
    # Open models
    "open-mistral-nemo": ModelPricing(
        model="open-mistral-nemo",
        provider="Mistral",
        input_price=Decimal("0.15"),
        output_price=Decimal("0.15"),
    ),
    "open-mixtral-8x22b": ModelPricing(
        model="open-mixtral-8x22b",
        provider="Mistral",
        input_price=Decimal("2.00"),
        output_price=Decimal("6.00"),
    ),
    # Embeddings
    "mistral-embed": ModelPricing(
        model="mistral-embed",
        provider="Mistral",
        input_price=Decimal("0.10"),
        output_price=None,
    ),
}

GROK_PRICING: Dict[str, ModelPricing] = {
    "grok-2": ModelPricing(
        model="grok-2",
        provider="Grok",
        input_price=Decimal("2.00"),
        output_price=Decimal("10.00"),
    ),
    "grok-2-1212": ModelPricing(
        model="grok-2-1212",
        provider="Grok",
        input_price=Decimal("2.00"),
        output_price=Decimal("10.00"),
    ),
    "grok-2-vision": ModelPricing(
        model="grok-2-vision",
        provider="Grok",
        input_price=Decimal("2.00"),
        output_price=Decimal("10.00"),
    ),
    "grok-2-vision-1212": ModelPricing(
        model="grok-2-vision-1212",
        provider="Grok",
        input_price=Decimal("2.00"),
        output_price=Decimal("10.00"),
    ),
    "grok-beta": ModelPricing(
        model="grok-beta",
        provider="Grok",
        input_price=Decimal("5.00"),
        output_price=Decimal("15.00"),
    ),
    "grok-vision-beta": ModelPricing(
        model="grok-vision-beta",
        provider="Grok",
        input_price=Decimal("5.00"),
        output_price=Decimal("15.00"),
    ),
}

# HuggingFace - typically free for hosted inference, costs for dedicated endpoints
HUGGINGFACE_PRICING: Dict[str, ModelPricing] = {
    "meta-llama/Llama-3.1-70B-Instruct": ModelPricing(
        model="meta-llama/Llama-3.1-70B-Instruct",
        provider="HuggingFace",
        input_price=Decimal("0.00"),
        output_price=Decimal("0.00"),
        notes="Free tier inference; dedicated endpoints vary",
    ),
    "meta-llama/Llama-3.1-8B-Instruct": ModelPricing(
        model="meta-llama/Llama-3.1-8B-Instruct",
        provider="HuggingFace",
        input_price=Decimal("0.00"),
        output_price=Decimal("0.00"),
    ),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ModelPricing(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        provider="HuggingFace",
        input_price=Decimal("0.00"),
        output_price=Decimal("0.00"),
    ),
}

# =============================================================================
# Image Generation Pricing
# =============================================================================

OPENAI_IMAGE_PRICING: Dict[str, ModelPricing] = {
    # DALL-E 3
    "dall-e-3": ModelPricing(
        model="dall-e-3",
        provider="OpenAI",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "1024x1024:standard": Decimal("0.040"),
            "1024x1024:hd": Decimal("0.080"),
            "1024x1792:standard": Decimal("0.080"),
            "1024x1792:hd": Decimal("0.120"),
            "1792x1024:standard": Decimal("0.080"),
            "1792x1024:hd": Decimal("0.120"),
        },
    ),
    # DALL-E 2
    "dall-e-2": ModelPricing(
        model="dall-e-2",
        provider="OpenAI",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "1024x1024": Decimal("0.020"),
            "512x512": Decimal("0.018"),
            "256x256": Decimal("0.016"),
        },
    ),
    # GPT-Image-1 (new 2025 model)
    "gpt-image-1": ModelPricing(
        model="gpt-image-1",
        provider="OpenAI",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "1024x1024:low": Decimal("0.011"),
            "1024x1024:medium": Decimal("0.042"),
            "1024x1024:high": Decimal("0.167"),
            "1024x1536:low": Decimal("0.016"),
            "1024x1536:medium": Decimal("0.063"),
            "1024x1536:high": Decimal("0.250"),
            "1536x1024:low": Decimal("0.016"),
            "1536x1024:medium": Decimal("0.063"),
            "1536x1024:high": Decimal("0.250"),
            "auto:low": Decimal("0.011"),
            "auto:medium": Decimal("0.042"),
            "auto:high": Decimal("0.167"),
        },
    ),
}

STABILITY_PRICING: Dict[str, ModelPricing] = {
    "stable-diffusion-xl-1024-v1-0": ModelPricing(
        model="stable-diffusion-xl-1024-v1-0",
        provider="Stability",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "1024x1024": Decimal("0.04"),
            "1024x1024:30_steps": Decimal("0.06"),
            "1024x1024:50_steps": Decimal("0.10"),
        },
        notes="Credit-based: ~$0.01 per credit",
    ),
    "stable-diffusion-3": ModelPricing(
        model="stable-diffusion-3",
        provider="Stability",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "1024x1024": Decimal("0.065"),
            "1536x1536": Decimal("0.10"),
        },
    ),
    "stable-diffusion-3-turbo": ModelPricing(
        model="stable-diffusion-3-turbo",
        provider="Stability",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "1024x1024": Decimal("0.04"),
        },
    ),
    "stable-image-ultra": ModelPricing(
        model="stable-image-ultra",
        provider="Stability",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "default": Decimal("0.08"),
        },
    ),
}

XAI_AURORA_PRICING: Dict[str, ModelPricing] = {
    "aurora": ModelPricing(
        model="aurora",
        provider="xAI",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "1024x1024": Decimal("0.07"),
            "1024x768": Decimal("0.07"),
            "768x1024": Decimal("0.07"),
        },
    ),
}

GOOGLE_IMAGEN_PRICING: Dict[str, ModelPricing] = {
    "imagen-3": ModelPricing(
        model="imagen-3",
        provider="Google",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "1024x1024": Decimal("0.04"),
            "default": Decimal("0.04"),
        },
    ),
    "imagen-3-fast": ModelPricing(
        model="imagen-3-fast",
        provider="Google",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "1024x1024": Decimal("0.02"),
            "default": Decimal("0.02"),
        },
    ),
}

BLACK_FOREST_LABS_PRICING: Dict[str, ModelPricing] = {
    "flux-pro": ModelPricing(
        model="flux-pro",
        provider="BlackForestLabs",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "default": Decimal("0.05"),
        },
    ),
    "flux-dev": ModelPricing(
        model="flux-dev",
        provider="BlackForestLabs",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "default": Decimal("0.025"),
        },
    ),
    "flux-schnell": ModelPricing(
        model="flux-schnell",
        provider="BlackForestLabs",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "default": Decimal("0.003"),
        },
    ),
}

IDEOGRAM_PRICING: Dict[str, ModelPricing] = {
    "ideogram-v2": ModelPricing(
        model="ideogram-v2",
        provider="Ideogram",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "default": Decimal("0.08"),
            "turbo": Decimal("0.05"),
        },
    ),
    "ideogram-v1-turbo": ModelPricing(
        model="ideogram-v1-turbo",
        provider="Ideogram",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "default": Decimal("0.04"),
        },
    ),
}

FAL_AI_PRICING: Dict[str, ModelPricing] = {
    "fal-flux-pro": ModelPricing(
        model="fal-flux-pro",
        provider="FalAI",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "default": Decimal("0.05"),
        },
    ),
    "fal-flux-dev": ModelPricing(
        model="fal-flux-dev",
        provider="FalAI",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "default": Decimal("0.025"),
        },
    ),
}

REPLICATE_PRICING: Dict[str, ModelPricing] = {
    "sdxl": ModelPricing(
        model="sdxl",
        provider="Replicate",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "default": Decimal("0.004"),
        },
        notes="Based on GPU seconds",
    ),
    "flux-schnell": ModelPricing(
        model="flux-schnell",
        provider="Replicate",
        input_price=Decimal("0"),
        unit=PricingUnit.PER_IMAGE,
        image_prices={
            "default": Decimal("0.003"),
        },
    ),
}

# =============================================================================
# Speech/Audio Pricing
# =============================================================================

OPENAI_AUDIO_PRICING: Dict[str, ModelPricing] = {
    "whisper-1": ModelPricing(
        model="whisper-1",
        provider="OpenAI",
        input_price=Decimal("0.006"),
        output_price=None,
        unit=PricingUnit.PER_MINUTE,
    ),
    "tts-1": ModelPricing(
        model="tts-1",
        provider="OpenAI",
        input_price=Decimal("15.00"),
        output_price=None,
        unit=PricingUnit.PER_1M_TOKENS,
        notes="$15/1M characters",
    ),
    "tts-1-hd": ModelPricing(
        model="tts-1-hd",
        provider="OpenAI",
        input_price=Decimal("30.00"),
        output_price=None,
        unit=PricingUnit.PER_1M_TOKENS,
    ),
}

ELEVENLABS_PRICING: Dict[str, ModelPricing] = {
    "eleven_multilingual_v2": ModelPricing(
        model="eleven_multilingual_v2",
        provider="ElevenLabs",
        input_price=Decimal("0.30"),
        output_price=None,
        unit=PricingUnit.PER_1K_CHARACTERS,
        notes="Varies by plan; Creator tier estimate",
    ),
    "eleven_turbo_v2": ModelPricing(
        model="eleven_turbo_v2",
        provider="ElevenLabs",
        input_price=Decimal("0.30"),
        output_price=None,
        unit=PricingUnit.PER_1K_CHARACTERS,
    ),
}


# =============================================================================
# Pricing Registry
# =============================================================================

# Module-level singleton
_pricing_registry_instance: Optional["PricingRegistry"] = None
_pricing_registry_lock: Optional[asyncio.Lock] = None


async def get_pricing_registry(
    config_manager: Optional["ConfigManager"] = None,
) -> "PricingRegistry":
    """Get the global PricingRegistry singleton.

    Args:
        config_manager: Optional configuration manager for custom pricing.

    Returns:
        Initialized PricingRegistry instance.
    """
    global _pricing_registry_instance, _pricing_registry_lock

    if _pricing_registry_instance is not None:
        return _pricing_registry_instance

    if _pricing_registry_lock is None:
        _pricing_registry_lock = asyncio.Lock()

    async with _pricing_registry_lock:
        if _pricing_registry_instance is None:
            _pricing_registry_instance = PricingRegistry(config_manager)
            await _pricing_registry_instance.initialize()
            logger.info("PricingRegistry singleton created")

    return _pricing_registry_instance


def get_pricing_registry_sync() -> Optional["PricingRegistry"]:
    """Get PricingRegistry if already initialized (non-async).

    Returns None if not yet initialized.
    """
    return _pricing_registry_instance


class PricingRegistry:
    """Central registry for provider and model pricing.

    Provides methods to look up current pricing and calculate costs
    for various operations across all supported providers.
    """

    def __init__(self, config_manager: Optional["ConfigManager"] = None):
        """Initialize the pricing registry.

        Args:
            config_manager: Optional configuration manager for overrides.
        """
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)

        # Combined pricing data
        self._pricing: Dict[str, ModelPricing] = {}

        # Provider to pricing map for lookups
        self._by_provider: Dict[str, Dict[str, ModelPricing]] = {}

        # Custom overrides from configuration
        self._custom_pricing: Dict[str, ModelPricing] = {}

        # Last update timestamp
        self._last_updated: Optional[datetime] = None

    async def initialize(self) -> None:
        """Initialize the registry with default pricing data."""
        self._load_default_pricing()
        await self._load_custom_pricing()
        self._last_updated = datetime.now(timezone.utc)
        self.logger.info(
            "Loaded pricing for %d models across %d providers",
            len(self._pricing),
            len(self._by_provider),
        )

    def _load_default_pricing(self) -> None:
        """Load all default pricing data."""
        pricing_sources = [
            ("OpenAI", OPENAI_PRICING),
            ("Anthropic", ANTHROPIC_PRICING),
            ("Google", GOOGLE_PRICING),
            ("Mistral", MISTRAL_PRICING),
            ("Grok", GROK_PRICING),
            ("HuggingFace", HUGGINGFACE_PRICING),
            ("OpenAI_Images", OPENAI_IMAGE_PRICING),
            ("Stability", STABILITY_PRICING),
            ("xAI", XAI_AURORA_PRICING),
            ("Google_Images", GOOGLE_IMAGEN_PRICING),
            ("BlackForestLabs", BLACK_FOREST_LABS_PRICING),
            ("Ideogram", IDEOGRAM_PRICING),
            ("FalAI", FAL_AI_PRICING),
            ("Replicate", REPLICATE_PRICING),
            ("OpenAI_Audio", OPENAI_AUDIO_PRICING),
            ("ElevenLabs", ELEVENLABS_PRICING),
        ]

        for provider, pricing_dict in pricing_sources:
            if provider not in self._by_provider:
                self._by_provider[provider] = {}

            for model_id, model_pricing in pricing_dict.items():
                self._pricing[model_id] = model_pricing
                self._by_provider[provider][model_id] = model_pricing

    async def _load_custom_pricing(self) -> None:
        """Load custom pricing overrides from configuration."""
        if self.config_manager is None:
            return

        try:
            custom = self.config_manager.get_config("BUDGET_CUSTOM_PRICING")
            if not isinstance(custom, dict):
                return

            for model_id, pricing_data in custom.items():
                if isinstance(pricing_data, dict):
                    model_pricing = ModelPricing(
                        model=model_id,
                        provider=pricing_data.get("provider", "Custom"),
                        input_price=Decimal(str(pricing_data.get("input_price", 0))),
                        output_price=Decimal(str(pricing_data.get("output_price", 0)))
                        if pricing_data.get("output_price") is not None
                        else None,
                    )
                    self._custom_pricing[model_id] = model_pricing
                    self._pricing[model_id] = model_pricing
                    self.logger.debug("Loaded custom pricing for %s", model_id)

        except Exception as exc:
            self.logger.warning("Failed to load custom pricing: %s", exc)

    def get_model_pricing(self, model: str) -> Optional[ModelPricing]:
        """Get pricing for a specific model.

        Args:
            model: Model identifier.

        Returns:
            ModelPricing if found, None otherwise.
        """
        # Check custom pricing first (allows overrides)
        if model in self._custom_pricing:
            return self._custom_pricing[model]

        # Check direct match
        if model in self._pricing:
            return self._pricing[model]

        # Try partial match (e.g., "gpt-4o" matches "gpt-4o-2024-08-06")
        for known_model in self._pricing:
            if model.startswith(known_model) or known_model.startswith(model):
                return self._pricing[known_model]

        return None

    def get_provider_models(self, provider: str) -> List[str]:
        """Get all models for a provider.

        Args:
            provider: Provider name.

        Returns:
            List of model identifiers.
        """
        provider_pricing = self._by_provider.get(provider, {})
        return list(provider_pricing.keys())

    def get_all_providers(self) -> List[str]:
        """Get list of all providers with pricing data."""
        return list(self._by_provider.keys())

    def calculate_llm_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> Decimal:
        """Calculate cost for an LLM request.

        Args:
            model: Model identifier.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            cached_tokens: Number of tokens served from cache.

        Returns:
            Calculated cost in USD.
        """
        pricing = self.get_model_pricing(model)
        if pricing is None:
            self.logger.warning("No pricing found for model: %s", model)
            return Decimal("0")

        # Calculate based on per-million pricing
        multiplier = Decimal("1000000")  # Prices are per 1M tokens

        # Non-cached input tokens
        regular_input = input_tokens - cached_tokens
        input_cost = (Decimal(str(regular_input)) / multiplier) * pricing.input_price

        # Cached input tokens (if pricing available)
        cached_cost = Decimal("0")
        if cached_tokens > 0 and pricing.cached_input_price is not None:
            cached_cost = (
                Decimal(str(cached_tokens)) / multiplier
            ) * pricing.cached_input_price

        # Output tokens
        output_cost = Decimal("0")
        if output_tokens > 0 and pricing.output_price is not None:
            output_cost = (Decimal(str(output_tokens)) / multiplier) * pricing.output_price

        total = _quantize(input_cost + cached_cost + output_cost)
        return total

    def calculate_image_cost(
        self,
        model: str,
        count: int = 1,
        size: str = "1024x1024",
        quality: str = "standard",
    ) -> Decimal:
        """Calculate cost for image generation.

        Args:
            model: Model identifier.
            count: Number of images generated.
            size: Image size (e.g., "1024x1024").
            quality: Image quality (e.g., "standard", "hd").

        Returns:
            Calculated cost in USD.
        """
        pricing = self.get_model_pricing(model)
        if pricing is None:
            self.logger.warning("No pricing found for image model: %s", model)
            return Decimal("0")

        # Build lookup key
        size_key = f"{size}:{quality}"
        price_per_image = pricing.image_prices.get(size_key)

        if price_per_image is None:
            # Try without quality
            price_per_image = pricing.image_prices.get(size)

        if price_per_image is None:
            # Try default
            price_per_image = pricing.image_prices.get("default", Decimal("0"))

        total = _quantize(price_per_image * Decimal(str(count)))
        return total

    def calculate_audio_cost(
        self,
        model: str,
        duration_seconds: float = 0,
        characters: int = 0,
    ) -> Decimal:
        """Calculate cost for audio operations.

        Args:
            model: Model identifier (whisper-1, tts-1, etc.).
            duration_seconds: Duration for STT operations.
            characters: Character count for TTS operations.

        Returns:
            Calculated cost in USD.
        """
        pricing = self.get_model_pricing(model)
        if pricing is None:
            self.logger.warning("No pricing found for audio model: %s", model)
            return Decimal("0")

        if pricing.unit == PricingUnit.PER_MINUTE:
            # STT pricing (per minute)
            minutes = Decimal(str(duration_seconds)) / Decimal("60")
            return _quantize(minutes * pricing.input_price)

        elif pricing.unit == PricingUnit.PER_1K_CHARACTERS:
            # TTS pricing (per 1K characters)
            k_chars = Decimal(str(characters)) / Decimal("1000")
            return _quantize(k_chars * pricing.input_price)

        elif pricing.unit == PricingUnit.PER_1M_TOKENS:
            # OpenAI TTS uses per-1M character pricing
            m_chars = Decimal(str(characters)) / Decimal("1000000")
            return _quantize(m_chars * pricing.input_price)

        return Decimal("0")

    def calculate_embedding_cost(
        self,
        model: str,
        tokens: int,
    ) -> Decimal:
        """Calculate cost for embedding generation.

        Args:
            model: Model identifier.
            tokens: Number of tokens embedded.

        Returns:
            Calculated cost in USD.
        """
        pricing = self.get_model_pricing(model)
        if pricing is None:
            self.logger.warning("No pricing found for embedding model: %s", model)
            return Decimal("0")

        # Per-million pricing
        m_tokens = Decimal(str(tokens)) / Decimal("1000000")
        return _quantize(m_tokens * pricing.input_price)

    def estimate_request_cost(
        self,
        provider: str,
        model: str,
        estimated_input_tokens: int = 0,
        estimated_output_tokens: int = 0,
        image_count: int = 0,
        image_size: str = "1024x1024",
        image_quality: str = "standard",
    ) -> Decimal:
        """Estimate cost before making a request.

        Useful for budget pre-checks.

        Args:
            provider: Provider name.
            model: Model identifier.
            estimated_input_tokens: Expected input tokens.
            estimated_output_tokens: Expected output tokens.
            image_count: Number of images to generate.
            image_size: Expected image size.
            image_quality: Expected image quality.

        Returns:
            Estimated cost in USD.
        """
        total = Decimal("0")

        if estimated_input_tokens > 0 or estimated_output_tokens > 0:
            total += self.calculate_llm_cost(
                model, estimated_input_tokens, estimated_output_tokens
            )

        if image_count > 0:
            total += self.calculate_image_cost(
                model, image_count, image_size, image_quality
            )

        return total

    def get_cheaper_alternative(
        self,
        model: str,
        provider: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        """Find a cheaper alternative to the given model.

        Args:
            model: Current model identifier.
            provider: Optionally limit to same provider.

        Returns:
            Tuple of (model, provider) if found, None otherwise.
        """
        current_pricing = self.get_model_pricing(model)
        if current_pricing is None:
            return None

        current_cost = current_pricing.input_price + (
            current_pricing.output_price or Decimal("0")
        )

        # Find cheaper alternatives
        alternatives: List[Tuple[str, str, Decimal]] = []

        search_providers = [provider] if provider else list(self._by_provider.keys())

        for prov in search_providers:
            if prov.endswith("_Images") or prov.endswith("_Audio"):
                continue  # Skip non-LLM providers

            for m_id, m_pricing in self._by_provider.get(prov, {}).items():
                if m_pricing.unit != PricingUnit.PER_1M_TOKENS:
                    continue
                if m_pricing.output_price is None:
                    continue  # Skip embeddings

                alt_cost = m_pricing.input_price + m_pricing.output_price
                if alt_cost < current_cost:
                    alternatives.append((m_id, m_pricing.provider, alt_cost))

        if not alternatives:
            return None

        # Return cheapest
        alternatives.sort(key=lambda x: x[2])
        return (alternatives[0][0], alternatives[0][1])

    def set_custom_pricing(
        self,
        model: str,
        provider: str,
        input_price: Decimal,
        output_price: Optional[Decimal] = None,
        image_prices: Optional[Dict[str, Decimal]] = None,
    ) -> None:
        """Set custom pricing for a model.

        Args:
            model: Model identifier.
            provider: Provider name.
            input_price: Input price per 1M tokens.
            output_price: Output price per 1M tokens.
            image_prices: Per-image pricing by size.
        """
        pricing = ModelPricing(
            model=model,
            provider=provider,
            input_price=input_price,
            output_price=output_price,
            image_prices=image_prices or {},
        )
        self._custom_pricing[model] = pricing
        self._pricing[model] = pricing
        self.logger.info("Set custom pricing for %s", model)

    def as_dict(self) -> Dict[str, Any]:
        """Export pricing data as dictionary."""
        return {
            "last_updated": self._last_updated.isoformat() if self._last_updated else None,
            "providers": list(self._by_provider.keys()),
            "model_count": len(self._pricing),
            "custom_overrides": list(self._custom_pricing.keys()),
        }
