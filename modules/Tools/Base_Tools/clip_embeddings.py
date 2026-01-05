"""CLIP Embeddings for Image Similarity and Semantic Search.

This module provides CLIP (Contrastive Language-Image Pre-training) based
embeddings for images and text, enabling:
- Image similarity search
- Text-to-image semantic search
- Image-to-text matching
- Cross-modal retrieval

Supports multiple backends:
- OpenAI CLIP API (recommended for production)
- Local transformers (requires accelerators)
- Replicate CLIP endpoints
"""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from modules.logging.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class EmbeddingResult:
    """Result from an embedding operation."""
    
    embedding: list[float]
    source: str  # "image" or "text"
    source_id: str  # Hash or identifier
    model: str
    dimensions: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimilarityResult:
    """Result from a similarity search."""
    
    id: str
    score: float  # Cosine similarity [-1, 1] normalized to [0, 1]
    source_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class CLIPEmbeddings:
    """CLIP-based embedding generator for images and text.
    
    Provides embeddings that can be used for:
    - Image-to-image similarity
    - Text-to-image search
    - Image-to-text matching
    
    Uses cosine similarity for comparing embeddings.
    """
    
    # Available backends
    BACKEND_OPENAI = "openai"
    BACKEND_LOCAL = "local"
    BACKEND_REPLICATE = "replicate"
    
    # Default model dimensions
    DIMENSIONS = {
        "openai": 1536,  # text-embedding-ada-002 compatible
        "clip-vit-base-patch32": 512,
        "clip-vit-large-patch14": 768,
        "clip-vit-large-patch14-336": 768,
    }
    
    def __init__(
        self,
        backend: str = BACKEND_OPENAI,
        model: str | None = None,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        """Initialize the CLIP embeddings handler.
        
        Args:
            backend: Backend to use ("openai", "local", "replicate")
            model: Specific model to use (backend-dependent)
            api_key: API key for cloud backends
            cache_dir: Directory for caching embeddings
        """
        self.backend = backend
        self.model = model or self._default_model(backend)
        self.api_key = api_key or self._get_api_key(backend)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._client: httpx.AsyncClient | None = None
        self._local_model: Any = None
        self._local_processor: Any = None
    
    def _default_model(self, backend: str) -> str:
        """Get default model for a backend."""
        if backend == self.BACKEND_OPENAI:
            return "text-embedding-3-small"  # OpenAI's embedding model
        elif backend == self.BACKEND_LOCAL:
            return "openai/clip-vit-base-patch32"
        elif backend == self.BACKEND_REPLICATE:
            return "andreasjansson/clip-features"
        return "clip-vit-base-patch32"
    
    def _get_api_key(self, backend: str) -> str | None:
        """Get API key from environment."""
        if backend == self.BACKEND_OPENAI:
            return os.environ.get("OPENAI_API_KEY")
        elif backend == self.BACKEND_REPLICATE:
            return os.environ.get("REPLICATE_API_TOKEN")
        return None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client
    
    def _compute_hash(self, data: bytes) -> str:
        """Compute hash for caching."""
        return hashlib.sha256(data).hexdigest()[:16]
    
    async def embed_image(
        self,
        image: str | bytes | Path,
        metadata: dict[str, Any] | None = None,
    ) -> EmbeddingResult:
        """Generate embedding for an image.
        
        Args:
            image: Image path, URL, or raw bytes
            metadata: Optional metadata to attach
            
        Returns:
            EmbeddingResult with the embedding vector
        """
        # Load image data
        image_data = await self._load_image(image)
        image_hash = self._compute_hash(image_data)
        
        # Check cache
        cached = self._load_from_cache(f"img_{image_hash}")
        if cached:
            return EmbeddingResult(
                embedding=cached["embedding"],
                source="image",
                source_id=image_hash,
                model=self.model,
                dimensions=len(cached["embedding"]),
                metadata=metadata or {},
            )
        
        # Generate embedding based on backend
        if self.backend == self.BACKEND_OPENAI:
            embedding = await self._embed_image_openai(image_data)
        elif self.backend == self.BACKEND_LOCAL:
            embedding = await self._embed_image_local(image_data)
        elif self.backend == self.BACKEND_REPLICATE:
            embedding = await self._embed_image_replicate(image_data)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        # Cache result
        self._save_to_cache(f"img_{image_hash}", {"embedding": embedding})
        
        return EmbeddingResult(
            embedding=embedding,
            source="image",
            source_id=image_hash,
            model=self.model,
            dimensions=len(embedding),
            metadata=metadata or {},
        )
    
    async def embed_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> EmbeddingResult:
        """Generate embedding for text.
        
        Args:
            text: Text to embed
            metadata: Optional metadata to attach
            
        Returns:
            EmbeddingResult with the embedding vector
        """
        text_hash = self._compute_hash(text.encode())
        
        # Check cache
        cached = self._load_from_cache(f"txt_{text_hash}")
        if cached:
            return EmbeddingResult(
                embedding=cached["embedding"],
                source="text",
                source_id=text_hash,
                model=self.model,
                dimensions=len(cached["embedding"]),
                metadata=metadata or {},
            )
        
        # Generate embedding
        if self.backend == self.BACKEND_OPENAI:
            embedding = await self._embed_text_openai(text)
        elif self.backend == self.BACKEND_LOCAL:
            embedding = await self._embed_text_local(text)
        elif self.backend == self.BACKEND_REPLICATE:
            embedding = await self._embed_text_replicate(text)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        # Cache result
        self._save_to_cache(f"txt_{text_hash}", {"embedding": embedding})
        
        return EmbeddingResult(
            embedding=embedding,
            source="text",
            source_id=text_hash,
            model=self.model,
            dimensions=len(embedding),
            metadata=metadata or {},
        )
    
    async def _load_image(self, image: str | bytes | Path) -> bytes:
        """Load image data from various sources."""
        if isinstance(image, bytes):
            return image
        
        if isinstance(image, Path):
            return image.read_bytes()
        
        if isinstance(image, str):
            # Check if it's a URL
            if image.startswith(("http://", "https://")):
                client = await self._get_client()
                response = await client.get(image)
                response.raise_for_status()
                return response.content
            
            # Check if it's base64
            if image.startswith("data:image"):
                # Extract base64 data
                _, data = image.split(",", 1)
                return base64.b64decode(data)
            
            # Assume it's a file path
            return Path(image).read_bytes()
        
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    async def _embed_image_openai(self, image_data: bytes) -> list[float]:
        """Embed image using OpenAI's vision capabilities."""
        # OpenAI doesn't have direct CLIP API, so we use a vision model
        # to describe the image and then embed the description
        # For true CLIP embeddings, use local or replicate backend
        
        client = await self._get_client()
        b64_image = base64.b64encode(image_data).decode()
        
        # First, get image description via vision model
        vision_response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in detail for embedding purposes. Include objects, colors, style, mood, and composition. Be concise but comprehensive.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 300,
            },
        )
        vision_response.raise_for_status()
        description = vision_response.json()["choices"][0]["message"]["content"]
        
        # Then embed the description
        return await self._embed_text_openai(description)
    
    async def _embed_text_openai(self, text: str) -> list[float]:
        """Embed text using OpenAI's embedding API."""
        client = await self._get_client()
        
        response = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "input": text,
            },
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    
    async def _embed_image_local(self, image_data: bytes) -> list[float]:
        """Embed image using local transformers."""
        self._ensure_local_model()
        
        import io
        from PIL import Image
        import torch
        
        image = Image.open(io.BytesIO(image_data))
        inputs = self._local_processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = self._local_model.get_image_features(**inputs)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.squeeze().tolist()
    
    async def _embed_text_local(self, text: str) -> list[float]:
        """Embed text using local transformers."""
        self._ensure_local_model()
        
        import torch
        
        inputs = self._local_processor(text=[text], return_tensors="pt", padding=True)
        
        with torch.no_grad():
            text_features = self._local_model.get_text_features(**inputs)
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.squeeze().tolist()
    
    def _ensure_local_model(self):
        """Lazy-load local transformers model."""
        if self._local_model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor
            except ImportError as e:
                raise ImportError(
                    "Local backend requires transformers. "
                    "Install with: pip install transformers torch"
                ) from e
            
            self._local_model = CLIPModel.from_pretrained(self.model)
            self._local_processor = CLIPProcessor.from_pretrained(self.model)
    
    async def _embed_image_replicate(self, image_data: bytes) -> list[float]:
        """Embed image using Replicate CLIP endpoint."""
        client = await self._get_client()
        b64_image = base64.b64encode(image_data).decode()
        
        response = await client.post(
            "https://api.replicate.com/v1/predictions",
            headers={"Authorization": f"Token {self.api_key}"},
            json={
                "version": self.model,
                "input": {
                    "image": f"data:image/jpeg;base64,{b64_image}",
                },
            },
        )
        response.raise_for_status()
        
        # Poll for result
        prediction = response.json()
        result_url = prediction["urls"]["get"]
        
        while True:
            result = await client.get(
                result_url,
                headers={"Authorization": f"Token {self.api_key}"},
            )
            result_data = result.json()
            if result_data["status"] == "succeeded":
                return result_data["output"]["embedding"]
            elif result_data["status"] == "failed":
                raise RuntimeError(f"Replicate prediction failed: {result_data.get('error')}")
            
            import asyncio
            await asyncio.sleep(0.5)
    
    async def _embed_text_replicate(self, text: str) -> list[float]:
        """Embed text using Replicate CLIP endpoint."""
        client = await self._get_client()
        
        response = await client.post(
            "https://api.replicate.com/v1/predictions",
            headers={"Authorization": f"Token {self.api_key}"},
            json={
                "version": self.model,
                "input": {
                    "text": text,
                },
            },
        )
        response.raise_for_status()
        
        # Poll for result
        prediction = response.json()
        result_url = prediction["urls"]["get"]
        
        while True:
            result = await client.get(
                result_url,
                headers={"Authorization": f"Token {self.api_key}"},
            )
            result_data = result.json()
            if result_data["status"] == "succeeded":
                return result_data["output"]["embedding"]
            elif result_data["status"] == "failed":
                raise RuntimeError(f"Replicate prediction failed: {result_data.get('error')}")
            
            import asyncio
            await asyncio.sleep(0.5)
    
    def _load_from_cache(self, key: str) -> dict | None:
        """Load embedding from cache."""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            import json
            return json.loads(cache_file.read_text())
        return None
    
    def _save_to_cache(self, key: str, data: dict):
        """Save embedding to cache."""
        if not self.cache_dir:
            return
        
        import json
        cache_file = self.cache_dir / f"{key}.json"
        cache_file.write_text(json.dumps(data))
    
    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.
        
        Args:
            a: First embedding vector
            b: Second embedding vector
            
        Returns:
            Cosine similarity in range [-1, 1]
        """
        import math
        
        if len(a) != len(b):
            raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    async def find_similar(
        self,
        query_embedding: list[float],
        candidates: list[EmbeddingResult],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SimilarityResult]:
        """Find similar embeddings to a query.
        
        Args:
            query_embedding: Query embedding vector
            candidates: List of candidate embeddings to search
            top_k: Maximum number of results
            min_score: Minimum similarity score (0 to 1)
            
        Returns:
            List of SimilarityResult sorted by score descending
        """
        results = []
        
        for candidate in candidates:
            score = self.cosine_similarity(query_embedding, candidate.embedding)
            # Normalize to [0, 1] range
            normalized_score = (score + 1) / 2
            
            if normalized_score >= min_score:
                results.append(SimilarityResult(
                    id=candidate.source_id,
                    score=normalized_score,
                    source_type=candidate.source,
                    metadata=candidate.metadata,
                ))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]
    
    async def text_to_image_search(
        self,
        query: str,
        image_embeddings: list[EmbeddingResult],
        top_k: int = 5,
    ) -> list[SimilarityResult]:
        """Search for images matching a text query.
        
        Args:
            query: Text query
            image_embeddings: Pre-computed image embeddings
            top_k: Maximum number of results
            
        Returns:
            List of matching images sorted by relevance
        """
        query_result = await self.embed_text(query)
        return await self.find_similar(
            query_result.embedding,
            image_embeddings,
            top_k=top_k,
        )
    
    async def image_to_image_search(
        self,
        query_image: str | bytes | Path,
        image_embeddings: list[EmbeddingResult],
        top_k: int = 5,
    ) -> list[SimilarityResult]:
        """Search for similar images.
        
        Args:
            query_image: Query image (path, URL, or bytes)
            image_embeddings: Pre-computed image embeddings
            top_k: Maximum number of results
            
        Returns:
            List of similar images sorted by similarity
        """
        query_result = await self.embed_image(query_image)
        return await self.find_similar(
            query_result.embedding,
            image_embeddings,
            top_k=top_k,
        )
    
    async def close(self):
        """Close HTTP client and free resources."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Module-level instance for convenience
_embeddings: CLIPEmbeddings | None = None


def get_clip_embeddings(
    backend: str = CLIPEmbeddings.BACKEND_OPENAI,
    **kwargs,
) -> CLIPEmbeddings:
    """Get or create a CLIPEmbeddings instance.
    
    Args:
        backend: Backend to use
        **kwargs: Additional arguments for CLIPEmbeddings
        
    Returns:
        CLIPEmbeddings instance
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = CLIPEmbeddings(backend=backend, **kwargs)
    return _embeddings
