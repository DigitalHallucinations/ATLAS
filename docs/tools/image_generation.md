---
audience: Persona authors and backend developers
status: active
last_verified: 2026-06-15
source_of_truth: modules/Providers/Media/base.py; modules/Providers/Media/manager.py; modules/Tools/Base_Tools/generate_image.py
---

# Image Generation Tools

ATLAS provides a unified interface for image generation across multiple AI providers. The media provider subsystem supports text-to-image generation, image editing (inpainting), image-to-image transformations, and prompt compilation for optimal results.

## Architecture Overview

The image generation system follows the same patterns as LLM providers:

```Text
┌─────────────────────────────────────────────────────────────────┐
│                    MediaProviderManager                          │
│  - Singleton orchestrator for all media providers               │
│  - Provider switching and caching                               │
│  - Health monitoring and fallback                               │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   OpenAI     │    │  Stability   │    │    FalAI     │
│   Images     │    │     AI       │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │ ImageGeneration  │
                    │     Result       │
                    └──────────────────┘
```

## Available Providers

| Provider | Environment Variable | Models | Features |
| --- | --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | dall-e-2, dall-e-3, gpt-image-1 | Text-to-image, inpainting, variations |
| Stability AI | `STABILITY_API_KEY` | stable-image-ultra, stable-image-core, sd3.5-*, sdxl | Text-to-image, inpainting, img2img |
| FalAI | `FAL_KEY` | flux-pro, flux-schnell, flux-dev | Fast inference, LoRA support |
| Google Imagen | `GOOGLE_API_KEY` | imagen-3, imagegeneration | Vertex AI integration |
| Black Forest Labs | `BFL_API_KEY` | flux-1.1-pro, flux-1.1-ultra | High-quality FLUX generation |
| XAI Aurora | `XAI_API_KEY` | grok-2-image | Grok-based generation |
| Hugging Face | `HUGGINGFACE_API_KEY` | Various diffusion models | Inference API or local |
| Replicate | `REPLICATE_API_TOKEN` | FLUX, SDXL, Kandinsky, many more | Open model aggregator |
| Ideogram | `IDEOGRAM_API_KEY` | V_2, V_2_TURBO, V_1 | Text-in-image specialist |
| Runway | `RUNWAY_API_KEY` | gen3a_turbo, gen2 | Gen-3 Alpha creative tools |

## Environment Variables

Configure provider access using environment variables:

```bash
# OpenAI (DALL·E)
OPENAI_API_KEY=sk-...

# Stability AI
STABILITY_API_KEY=sk-...

# FalAI
FAL_KEY=...

# Google Vertex AI / Imagen
GOOGLE_API_KEY=...
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GOOGLE_CLOUD_PROJECT=your-project-id

# Black Forest Labs
BFL_API_KEY=...

# XAI (Grok)
XAI_API_KEY=xai-...

# Hugging Face
HUGGINGFACE_API_KEY=hf_...

# Replicate
REPLICATE_API_TOKEN=r8_...

# Ideogram
IDEOGRAM_API_KEY=...

# Runway
RUNWAY_API_KEY=...
```

## Request Parameters

The `ImageGenerationRequest` dataclass accepts:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `prompt` | str | Required | Text description of the image to generate |
| `model` | str | Provider default | Model identifier (provider-specific) |
| `n` | int | 1 | Number of images to generate (1-4) |
| `size` | str | "1024x1024" | Image dimensions |
| `aspect_ratio` | str | None | Alternative to size (e.g., "16:9") |
| `style_preset` | str | None | Style preset name (provider-specific) |
| `quality` | str | "standard" | Quality level: draft, standard, hd |
| `seed` | int | None | Random seed for reproducibility |
| `input_images` | List[str] | None | Paths/URLs for img2img operations |
| `mask_image` | str | None | Mask path for inpainting |
| `output_format` | OutputFormat | FILEPATH | url, b64, or filepath |
| `safety` | Dict | None | Provider-specific safety settings |
| `metadata` | Dict | {} | Trace info (conversation_id, persona) |

### Extended Metadata Fields

Additional parameters can be passed via the `metadata` dictionary:

```python
request = ImageGenerationRequest(
    prompt="A serene mountain landscape",
    model="stable-diffusion-xl-1024-v1-0",
    metadata={
        "negative_prompt": "blurry, low quality",
        "guidance_scale": 7.5,
        "num_inference_steps": 30,
        "conversation_id": "conv-123",
        "persona": "Artist"
    }
)
```

## Response Format

The `ImageGenerationResult` contains:

| Field | Type | Description |
| --- | --- | --- |
| `success` | bool | Whether generation succeeded |
| `images` | List[GeneratedImage] | Generated images |
| `provider` | str | Provider name used |
| `model` | str | Model identifier used |
| `timing_ms` | int | Generation duration in milliseconds |
| `cost_estimate` | float | Optional cost estimate |
| `error` | str | Error message if failed |

Each `GeneratedImage` includes:

| Field | Type | Description |
| --- | --- | --- |
| `id` | str | Unique identifier |
| `mime` | str | MIME type (e.g., "image/png") |
| `path` | str | Local filesystem path |
| `url` | str | Remote URL if available |
| `b64` | str | Base64-encoded data |
| `seed_used` | int | Random seed used |
| `revised_prompt` | str | Model-revised prompt |

## Tool Usage

### generate_image

Basic text-to-image generation:

```json
{
    "tool": "generate_image",
    "parameters": {
        "prompt": "A futuristic cityscape at sunset",
        "provider": "openai",
        "model": "dall-e-3",
        "size": "1792x1024",
        "quality": "hd"
    }
}
```

### edit_image

Image editing with inpainting:

```json
{
    "tool": "edit_image",
    "parameters": {
        "prompt": "Add a rainbow in the sky",
        "input_image": "/path/to/image.png",
        "mask_image": "/path/to/mask.png",
        "provider": "stability"
    }
}
```

### prompt_compiler

Enhance prompts for better generation:

```json
{
    "tool": "prompt_compiler",
    "parameters": {
        "base_prompt": "cat",
        "style": "photorealistic",
        "mood": "peaceful",
        "lighting": "golden hour"
    }
}
```

### clip_embeddings

Generate CLIP embeddings for semantic search:

```json
{
    "tool": "clip_embeddings",
    "parameters": {
        "image_path": "/path/to/image.png"
    }
}
```

## Configuration

Configure media providers in `atlas_config.yaml`:

```yaml
media:
  default_provider: openai
  providers:
    openai:
      enabled: true
      default_model: dall-e-3
      default_size: "1024x1024"
    stability:
      enabled: true
      default_model: stable-diffusion-xl-1024-v1-0
    falai:
      enabled: true
      default_model: flux-pro
  output:
    save_to_disk: true
    base_path: "assets/generated"
    format: png
  safety:
    block_nsfw: true
    content_filter: moderate
```

## Provider-Specific Notes

### OpenAI (DALL·E)

- DALL·E 3 supports `1024x1024`, `1792x1024`, `1024x1792`
- Quality options: `standard`, `hd`
- Automatically revises prompts for safety

### Stability AI

- Supports negative prompts via metadata
- Guidance scale and inference steps configurable
- Multiple engine versions available

### FalAI

- Fast inference with Flux models
- Supports LoRA model mixing
- Queue-based async generation

### Google Imagen

- Requires Vertex AI project setup
- Uses service account authentication
- Supports aspect ratios

### Black Forest Labs

- High-quality Flux models (FLUX.1 Pro, Ultra, Dev, Schnell)
- Supports ultra-high resolution
- Async polling for results
- Direct API access (vs aggregators like FalAI)

### Replicate

- Access to thousands of open-source models
- Supports FLUX, SDXL, Kandinsky, Playground, and more
- Pay-per-second GPU pricing
- Custom model versions via model:version format

### Ideogram

- Specialized in accurate text rendering within images
- Ideal for logos, posters, and marketing materials
- Magic prompt enhancement option
- Multiple style types (realistic, design, anime, 3D)

### Runway

- Gen-3 Alpha cutting-edge generation
- Text-to-image and image-to-image
- Async task-based processing
- Video generation capabilities (separate API)

## Error Handling

The media provider system handles failures gracefully:

```python
result = await manager.generate_image(request)
if not result.success:
    logger.error(f"Generation failed: {result.error}")
    # Fallback logic or retry
```

Common error scenarios:

- API key invalid or missing
- Rate limiting
- Content policy violations
- Network timeouts
- Model unavailable

## Cost Tracking

Each generation can include cost estimates:

```python
result = await manager.generate_image(request)
if result.cost_estimate:
    logger.info(f"Estimated cost: ${result.cost_estimate:.4f}")
```

## Related Documentation

- [Tool Manifest Metadata](../tool-manifest.md) - Tool configuration and validation
- [Configuration Reference](../configuration.md) - YAML/env configuration
- [Vector Store Tool](vector_store.md) - For storing image embeddings
