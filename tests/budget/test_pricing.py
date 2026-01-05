"""Tests for pricing registry and cost calculation."""

from __future__ import annotations

from decimal import Decimal

import pytest

from modules.budget.pricing import (
    ModelPricing,
    PricingRegistry,
    PricingUnit,
    get_pricing_registry,
    OPENAI_PRICING,
    ANTHROPIC_PRICING,
)


class TestModelPricing:
    """Tests for ModelPricing dataclass."""

    def test_create_llm_pricing(self) -> None:
        """Test creating LLM model pricing."""
        pricing = ModelPricing(
            model="test-model",
            provider="TestProvider",
            input_price=Decimal("2.50"),
            output_price=Decimal("10.00"),
            unit=PricingUnit.PER_1M_TOKENS,
        )

        assert pricing.model == "test-model"
        assert pricing.provider == "TestProvider"
        assert pricing.input_price == Decimal("2.50")
        assert pricing.output_price == Decimal("10.00")
        assert pricing.unit == PricingUnit.PER_1M_TOKENS

    def test_create_image_pricing(self) -> None:
        """Test creating image model pricing."""
        pricing = ModelPricing(
            model="dall-e-3",
            provider="OpenAI",
            input_price=Decimal("0"),
            unit=PricingUnit.PER_IMAGE,
            image_prices={
                "1024x1024": Decimal("0.040"),
                "1024x1792": Decimal("0.080"),
            },
        )

        assert pricing.unit == PricingUnit.PER_IMAGE
        assert "1024x1024" in pricing.image_prices
        assert pricing.image_prices["1024x1024"] == Decimal("0.040")

    def test_pricing_decimal_normalization(self) -> None:
        """Test that numeric values are normalized to Decimal."""
        pricing = ModelPricing(
            model="normalize-test",
            provider="Test",
            input_price=2.5,  # type: ignore[arg-type]  # Testing runtime coercion
            output_price=10,  # type: ignore[arg-type]  # Testing runtime coercion
        )

        assert isinstance(pricing.input_price, Decimal)
        assert isinstance(pricing.output_price, Decimal)

    def test_pricing_with_cached_input(self) -> None:
        """Test pricing with cached input price."""
        pricing = ModelPricing(
            model="cached-model",
            provider="Test",
            input_price=Decimal("5.00"),
            output_price=Decimal("15.00"),
            cached_input_price=Decimal("2.50"),
        )

        assert pricing.cached_input_price == Decimal("2.50")


class TestPricingRegistry:
    """Tests for PricingRegistry singleton."""

    @pytest.mark.asyncio
    async def test_get_pricing_registry_singleton(self) -> None:
        """Test that get_pricing_registry returns the same instance."""
        registry1 = await get_pricing_registry()
        registry2 = await get_pricing_registry()

        assert registry1 is registry2

    @pytest.mark.asyncio
    async def test_registry_has_openai_models(self) -> None:
        """Test that registry includes OpenAI models."""
        registry = await get_pricing_registry()

        pricing = registry.get_model_pricing("gpt-4o")
        assert pricing is not None
        assert pricing.provider == "OpenAI"

    @pytest.mark.asyncio
    async def test_registry_has_anthropic_models(self) -> None:
        """Test that registry includes Anthropic models."""
        registry = await get_pricing_registry()

        # Use actual model name from pricing dict
        pricing = registry.get_model_pricing("claude-3-5-sonnet-20241022")
        assert pricing is not None
        assert pricing.provider == "Anthropic"

    @pytest.mark.asyncio
    async def test_registry_unknown_model_returns_none(self) -> None:
        """Test that unknown models return None."""
        registry = await get_pricing_registry()

        pricing = registry.get_model_pricing("nonexistent-model-xyz")
        assert pricing is None

    @pytest.mark.asyncio
    async def test_calculate_llm_cost(self) -> None:
        """Test LLM cost calculation."""
        registry = await get_pricing_registry()

        # GPT-4o: $2.50 input, $10.00 output per 1M tokens
        cost = registry.calculate_llm_cost(
            model="gpt-4o",
            input_tokens=1000,  # 0.001M tokens
            output_tokens=500,  # 0.0005M tokens
        )

        # Expected: (1000/1M * 2.50) + (500/1M * 10.00)
        #         = 0.0025 + 0.005 = 0.0075
        assert cost is not None
        assert cost == pytest.approx(Decimal("0.0075"), rel=Decimal("0.0001"))

    @pytest.mark.asyncio
    async def test_calculate_llm_cost_with_cached_tokens(self) -> None:
        """Test LLM cost with cached tokens."""
        registry = await get_pricing_registry()

        # GPT-4o has cached pricing
        cost = registry.calculate_llm_cost(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=500,
        )

        # Cost should be less than without caching
        cost_without_cache = registry.calculate_llm_cost(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=0,
        )

        assert cost is not None
        assert cost_without_cache is not None
        assert cost < cost_without_cache

    @pytest.mark.asyncio
    async def test_calculate_llm_cost_unknown_model_returns_zero(self) -> None:
        """Test cost calculation for unknown model returns zero."""
        registry = await get_pricing_registry()

        cost = registry.calculate_llm_cost(
            model="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )

        # Returns Decimal("0") for unknown models
        assert cost == Decimal("0")

    @pytest.mark.asyncio
    async def test_calculate_image_cost(self) -> None:
        """Test image generation cost calculation."""
        registry = await get_pricing_registry()

        cost = registry.calculate_image_cost(
            model="dall-e-3",
            size="1024x1024",
            count=2,
        )

        assert cost is not None
        # DALL-E 3 1024x1024 standard: $0.040 per image
        assert cost == pytest.approx(Decimal("0.080"), rel=Decimal("0.001"))

    @pytest.mark.asyncio
    async def test_calculate_image_cost_hd_quality(self) -> None:
        """Test image cost with HD quality."""
        registry = await get_pricing_registry()

        cost_standard = registry.calculate_image_cost(
            model="dall-e-3",
            size="1024x1024",
        )
        cost_hd = registry.calculate_image_cost(
            model="dall-e-3",
            size="1024x1024",
            quality="hd",
        )

        assert cost_hd is not None
        assert cost_standard is not None
        assert cost_hd > cost_standard

    @pytest.mark.asyncio
    async def test_calculate_embedding_cost(self) -> None:
        """Test embedding cost calculation."""
        registry = await get_pricing_registry()

        cost = registry.calculate_embedding_cost(
            model="text-embedding-3-small",
            tokens=10000,
        )

        assert cost is not None
        # text-embedding-3-small: $0.02 per 1M tokens
        # 10000 tokens = 0.01M tokens = $0.0002
        assert cost == pytest.approx(Decimal("0.0002"), rel=Decimal("0.0001"))

    @pytest.mark.asyncio
    async def test_get_provider_models(self) -> None:
        """Test getting all models for a provider."""
        registry = await get_pricing_registry()

        openai_models = registry.get_provider_models("OpenAI")
        assert len(openai_models) > 0

    @pytest.mark.asyncio
    async def test_get_all_providers(self) -> None:
        """Test getting list of all providers."""
        registry = await get_pricing_registry()

        providers = registry.get_all_providers()
        assert "OpenAI" in providers
        assert "Anthropic" in providers
        assert "Google" in providers


class TestBuiltInPricingData:
    """Tests for built-in pricing dictionaries."""

    def test_openai_pricing_has_gpt4o(self) -> None:
        """Test OpenAI pricing includes GPT-4o models."""
        assert "gpt-4o" in OPENAI_PRICING
        assert "gpt-4o-mini" in OPENAI_PRICING

    def test_anthropic_pricing_has_claude(self) -> None:
        """Test Anthropic pricing includes Claude models."""
        # Use actual model names from the dict
        assert "claude-3-5-sonnet-20241022" in ANTHROPIC_PRICING
        assert "claude-3-opus-20240229" in ANTHROPIC_PRICING

    def test_pricing_values_are_positive(self) -> None:
        """Test all pricing values are positive."""
        for model, pricing in OPENAI_PRICING.items():
            assert pricing.input_price >= 0, f"{model} has negative input price"
            if pricing.output_price is not None:
                assert pricing.output_price >= 0, f"{model} has negative output price"

        for model, pricing in ANTHROPIC_PRICING.items():
            assert pricing.input_price >= 0, f"{model} has negative input price"
            if pricing.output_price is not None:
                assert pricing.output_price >= 0, f"{model} has negative output price"
