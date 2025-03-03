# modules/Hybrid_Cognitive_Dynamics_Model/Memory/Long_Term/Semantic/long_term_semantic_memory.py

import asyncio
import numpy as np
import networkx as nx
from typing import List, Set, Tuple, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import GaussianRandomProjection
from modules.Config.config import ConfigManager

class EnhancedLongTermSemanticMemory:
    def __init__(self, config_manager = ConfigManager, n_components: Optional[int] = None):
        self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('EnhancedLongTermSemanticMemory')
        self.logger.info("EnhancedLongTermSemanticMemory init in progress")

        memory_config = self.config_manager.get_subsystem_config('memory')
        
        # Use config values if provided, otherwise fall back to defaults
        self.n_components = n_components or memory_config.get('long_term_semantic', {}).get('max_concepts', 1000)
        self.pattern_completion_threshold = memory_config.get('consolidation', {}).get('pattern_completion_threshold', 0.8)

        self.knowledge_graph = nx.Graph()
        self.vectorizer = TfidfVectorizer(max_features=1000)

        self.n_components = min(self.n_components, 1000)
        self.random_projection = None

        self.memory_vectors = {}
        self.logger.debug(f"Initialized EnhancedLongTermSemanticMemory with {self.n_components} components")

    async def add(self, concept: str, related_concepts: List[str]) -> None:
        self.knowledge_graph.add_node(concept)
        for related in related_concepts:
            self.knowledge_graph.add_edge(concept, related)

        all_concepts = list(self.knowledge_graph.nodes)
        self.vectorizer.fit(all_concepts)

        concept_vector = self.vectorizer.transform([concept]).toarray()

        if self.random_projection is None:
            n_features = concept_vector.shape[1]
            self.n_components = min(self.n_components, n_features)
            self.random_projection = GaussianRandomProjection(n_components=self.n_components)
            self.logger.debug(f"Initialized GaussianRandomProjection with {self.n_components} components")

        projected_vector = self.random_projection.fit_transform(concept_vector)
        self.memory_vectors[concept] = projected_vector
        self.logger.debug(f"Added concept to EnhancedLongTermSemanticMemory: {concept}")

    async def pattern_completion(self, partial_concept: str, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        try:
            threshold = threshold or self.pattern_completion_threshold

            if not hasattr(self.vectorizer, 'vocabulary_'):
                self.vectorizer.fit([partial_concept])
                return []

            partial_vector = self.vectorizer.transform([partial_concept]).toarray()
            projected_partial = self.random_projection.transform(partial_vector)

            completed_concepts = []
            for concept, vector in self.memory_vectors.items():
                similarity = cosine_similarity(projected_partial, vector)[0][0]
                if similarity > threshold:
                    completed_concepts.append((concept, float(similarity)))

            return sorted(completed_concepts, key=lambda x: x[1], reverse=True)
        except Exception as e:
            self.logger.error(f"Error in pattern_completion: {str(e)}")
            return []

    async def query(self, concept: str, n: int = 5) -> List[Tuple[str, float]]:
        try:
            if concept not in self.knowledge_graph:
                return []

            neighbors = list(self.knowledge_graph.neighbors(concept))
            if not neighbors:
                return []

            texts = [concept] + neighbors
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            similar_concepts = sorted(zip(neighbors, cosine_similarities), key=lambda x: x[1], reverse=True)
            return [(str(c), float(s)) for c, s in similar_concepts[:n]]
        except Exception as e:
            self.logger.error(f"Error in semantic memory query: {str(e)}", exc_info=True)
            return []

    def _extract_concepts(self, text: str) -> Set[str]:
        if not hasattr(self.vectorizer, 'vocabulary_'):
            self.vectorizer.fit([text])
        
        tfidf_matrix = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        
        important_terms = set()
        for _, idx in zip(*tfidf_matrix.nonzero()):
            important_terms.add(feature_names[idx])
        
        bigrams = self._extract_bigrams(text)
        important_terms.update(bigrams)
        
        return important_terms

    def _extract_bigrams(self, text: str) -> Set[str]:
        words = text.split()
        return set(' '.join(words[i:i+2]) for i in range(len(words)-1))

    async def preload_LTMS(self, data: List[str]):
        self.logger.info("Preloading Long Term Memory Store into EnhancedLongTermSemanticMemory")
        for text in data:
            concepts = self._extract_concepts(text)
            for concept in concepts:
                related_concepts = list(concepts - {concept})
                await self.add(concept, related_concepts)
            self.logger.info(f"Processed text: {text[:50]}...")
        self.logger.info(f"Preloaded {len(data)} items into semantic memory")

    async def process_input(self, input_data: Any) -> None:
        if isinstance(input_data, str):
            concepts = self._extract_concepts(input_data)
            for concept in concepts:
                related_concepts = list(concepts - {concept})
                await self.add(concept, related_concepts)
        elif isinstance(input_data, dict):
            for key, value in input_data.items():
                if isinstance(value, str):
                    await self.process_input(value)
        self.logger.debug(f"Processed input: {input_data[:50]}...")

    async def retrieve_relevant_info(self, query: str, n: int = 5) -> List[Tuple[str, float]]:
        completed_patterns = await self.pattern_completion(query)
        query_results = self.query(query, n)
        
        combined_results = completed_patterns + query_results
        sorted_results = sorted(combined_results, key=lambda x: x[1], reverse=True)
        
        return sorted_results[:n]

    def get_memory_stats(self) -> Dict[str, int]:
        return {
            "num_concepts": len(self.knowledge_graph.nodes()),
            "num_relationships": len(self.knowledge_graph.edges()),
        }

    async def update_from_consciousness(self, thought: Dict[str, Any]) -> None:
        if 'content' in thought:
            await self.process_input(thought['content'])
        self.logger.debug(f"Updated from consciousness: {thought.get('type', 'Unknown thought type')}")

    async def consolidate_memory(self) -> None:
        """
        Consolidate semantic memory by optimizing the knowledge graph and memory vectors.
        """
        self.logger.info("Performing semantic memory consolidation")
        
        # Prune rarely accessed or low-importance nodes
        self._prune_knowledge_graph()
        
        # Merge similar concepts
        self._merge_similar_concepts()
        
        # Update memory vectors based on recent interactions
        await self._update_memory_vectors()
        
        # Reorganize the knowledge graph for optimal retrieval
        self._reorganize_knowledge_graph()
        
        self.logger.info(f"Semantic memory consolidation completed. Graph nodes: {len(self.knowledge_graph.nodes())}, Memory vectors: {len(self.memory_vectors)}")

    def _prune_knowledge_graph(self):
        """Remove nodes with low importance or low connectivity."""
        nodes_to_remove = []
        for node in self.knowledge_graph.nodes():
            if self.knowledge_graph.degree(node) < 2 and node not in self.memory_vectors:
                nodes_to_remove.append(node)
        self.knowledge_graph.remove_nodes_from(nodes_to_remove)
        self.logger.debug(f"Pruned {len(nodes_to_remove)} nodes from the knowledge graph")

    def _merge_similar_concepts(self):
        """Merge concepts that are very similar to reduce redundancy."""
        merged_concepts = {}
        for concept1 in list(self.knowledge_graph.nodes()):
            for concept2 in list(self.knowledge_graph.nodes()):
                if concept1 != concept2 and concept1 not in merged_concepts and concept2 not in merged_concepts:
                    if self._compute_concept_similarity(concept1, concept2) > 0.9:  # High similarity threshold
                        merged_concept = f"{concept1}_{concept2}"
                        self._merge_nodes(concept1, concept2, merged_concept)
                        merged_concepts[concept1] = merged_concept
                        merged_concepts[concept2] = merged_concept
        self.logger.debug(f"Merged {len(merged_concepts)} pairs of similar concepts")

    async def _update_memory_vectors(self):
        """Update memory vectors based on recent interactions and current knowledge graph."""
        for concept in self.memory_vectors:
            related_concepts = list(self.knowledge_graph.neighbors(concept))
            if related_concepts:
                # Use the language model to generate an updated embedding
                updated_vector = await self._generate_concept_embedding(concept, related_concepts)
                self.memory_vectors[concept] = updated_vector
        self.logger.debug(f"Updated {len(self.memory_vectors)} memory vectors")

    def _reorganize_knowledge_graph(self):
        """Reorganize the knowledge graph for optimal retrieval."""
        # Implement graph reorganization logic
        # This could involve creating new connections, adjusting edge weights, etc.
        # For simplicity, we'll just ensure all nodes are connected
        components = list(nx.connected_components(self.knowledge_graph))
        if len(components) > 1:
            for i in range(1, len(components)):
                node1 = next(iter(components[0]))
                node2 = next(iter(components[i]))
                self.knowledge_graph.add_edge(node1, node2)
        self.logger.debug("Reorganized knowledge graph")

    def _compute_concept_similarity(self, concept1: str, concept2: str) -> float:
        """Compute similarity between two concepts."""
        if concept1 in self.memory_vectors and concept2 in self.memory_vectors:
            return cosine_similarity(
                self.memory_vectors[concept1].reshape(1, -1),
                self.memory_vectors[concept2].reshape(1, -1)
            )[0][0]
        return 0.0

    def _merge_nodes(self, node1: str, node2: str, new_node: str):
        """Merge two nodes in the knowledge graph."""
        self.knowledge_graph.add_node(new_node)
        for neighbor in set(self.knowledge_graph.neighbors(node1)) | set(self.knowledge_graph.neighbors(node2)):
            self.knowledge_graph.add_edge(new_node, neighbor)
        self.knowledge_graph.remove_node(node1)
        self.knowledge_graph.remove_node(node2)
        if node1 in self.memory_vectors and node2 in self.memory_vectors:
            self.memory_vectors[new_node] = (self.memory_vectors[node1] + self.memory_vectors[node2]) / 2
            del self.memory_vectors[node1]
            del self.memory_vectors[node2]

    async def _generate_concept_embedding(self, concept: str, related_concepts: List[str]) -> np.ndarray:
        """Generate a new embedding for a concept using the language model."""
        prompt = f"Concept: {concept}\nRelated concepts: {', '.join(related_concepts)}\nProvide a comprehensive description of this concept."
        response = await self.provider_manager.generate_response(
            messages=[{"role": "user", "content": prompt}],
            llm_call_type="semantic_consolidation"
        )
        # Assuming self.vectorizer is set up to generate embeddings
        return self.vectorizer.transform([response]).toarray()[0]

    async def start_periodic_consolidation(self):
        while True:
            await asyncio.sleep(self.consolidation_interval)
            await self.consolidate_memory()
