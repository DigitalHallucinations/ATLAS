# modules/Hybrid_Cognitive_Dynamics_Model/state_measurement.py  

import threading
import numpy as np
import time
import json
import re
from textblob import TextBlob
from modules.Config.config import ConfigManager
from modules.Providers.provider_manager import ProviderManager

# Singleton decorator
def singleton(cls):
    """
    Singleton decorator to apply thread-safe Singleton pattern.
    """
    cls._instance_lock = threading.Lock()

    def wrapper(*args, **kwargs):
        with cls._instance_lock:
            if not hasattr(cls, '_instance'):
                cls._instance = cls(*args, **kwargs)
        return cls._instance

    return wrapper

@singleton
class StateMeasurement:
    def __init__(self, provider_manager: ProviderManager, config_manager: ConfigManager = None):
        if config_manager is None:
            self.config_manager = ConfigManager()
        else:
            self.config_manager = config_manager
        self.logger = self.config_manager.setup_logger('StateMeasurement')
        self.logger.info("Initializing StateMeasurement")
        self.provider_manager = provider_manager

        # Use settings from ConfigManager
        state_measurement_config = self.config_manager.get_subsystem_config('state_measurement') if self.config_manager else {}
        self.default_analysis_method = state_measurement_config.get('default_analysis_method', 'textblob')

        self.logger.info("StateMeasurement initialized successfully")

    async def analyze_text(self, text: str) -> np.ndarray:
        start_time = time.time()
        try:
            self.logger.debug(f"Starting analysis for text: {text}")

            topic, topic_focus = await self._get_topic_classification(text)
            emotional_valence = await self._get_sentiment_analysis(text)
            arousal, dominance = self._get_additional_metrics(text)
            attention_allocation = self._compute_attention_allocation(text)
            memory_activation = await self._get_memory_activation(text)
            cognitive_control, consciousness_level, cognitive_load, mental_energy = self._compute_cognitive_metrics(text, dominance, topic_focus, attention_allocation)
            curiosity_level, uncertainty, creativity_index = self._compute_creativity_metrics(text, topic_focus)

            measurement_vector = self._assemble_measurement_vector(
                topic_focus,
                emotional_valence,
                arousal,
                dominance,
                attention_allocation,
                memory_activation,
                cognitive_control,
                consciousness_level,
                cognitive_load,
                mental_energy,
                curiosity_level,
                uncertainty,
                creativity_index
            )

            self.logger.debug(f"Final measurement vector: {measurement_vector}")

            return measurement_vector

        except Exception as e:
            self.logger.error(f"Error in analyze_text: {str(e)}", exc_info=True)
            self.logger.debug("Returning default zero-filled measurement vector")
            return np.zeros(108)

        finally:
            end_time = time.time()
            self.logger.info(f"State measurement completed in {end_time - start_time:.2f} seconds")

    async def derive_measurement(self, model_output: str, user_input: str) -> np.ndarray:
        combined_text = f"{user_input} {model_output}"
        self.logger.debug(f"Deriving measurement from combined text: {combined_text}")
        return await self.analyze_text(combined_text)

    async def get_topic_distribution(self, text: str) -> dict:
        try:
            self.logger.debug("Requesting topic distribution from LLM.")
            topic_result = await self.provider_manager.generate_response(
                messages=[
                    {"role": "system", "content": "Analyze the following text and provide a distribution of topics. Respond with a JSON object where keys are topic names and values are probabilities (0-1). The sum of all probabilities should be 1."},
                    {"role": "user", "content": text}
                ],
                llm_call_type="state_measurement"
            )
            distribution = json.loads(topic_result)
            self.logger.debug(f"Received topic distribution: {distribution}")
            return distribution
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON format in topic distribution response: '{topic_result}'")
            return {"error": 1.0}
        except Exception as e:
            self.logger.error(f"Error in get_topic_distribution: {str(e)}", exc_info=True)
            return {"error": 1.0}

    async def get_emotional_spectrum(self, text: str) -> dict:
        try:
            self.logger.debug("Requesting emotional spectrum from LLM.")
            emotion_result = await self.provider_manager.generate_response(
                messages=[
                    {"role": "system", "content": "Analyze the emotional spectrum of the following text. Respond with a JSON object where keys are emotion names and values are intensities (0-1)."},
                    {"role": "user", "content": text}
                ],
                llm_call_type="state_measurement"
            )
            spectrum = json.loads(emotion_result)
            self.logger.debug(f"Received emotional spectrum: {spectrum}")
            return spectrum
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON format in emotional spectrum response: '{emotion_result}'")
            return {"neutral": 1.0}
        except Exception as e:
            self.logger.error(f"Error in get_emotional_spectrum: {str(e)}", exc_info=True)
            return {"neutral": 1.0}

    async def estimate_cognitive_complexity(self, text: str) -> float:
        try:
            self.logger.debug("Requesting cognitive complexity estimate from LLM.")
            complexity_result = await self.provider_manager.generate_response(
                messages=[
                    {"role": "system", "content": "Estimate the cognitive complexity of the following text on a scale of 0 to 1, where 0 is very simple and 1 is extremely complex. Respond with a single float value."},
                    {"role": "user", "content": text}
                ],
                llm_call_type="state_measurement"
            )
            complexity = float(complexity_result.strip())
            self.logger.debug(f"Received cognitive complexity: {complexity}")
            return complexity
        except ValueError:
            self.logger.error(f"Invalid cognitive complexity format: '{complexity_result}'")
            return 0.5  # Return a neutral complexity if format is invalid
        except Exception as e:
            self.logger.error(f"Error in estimate_cognitive_complexity: {str(e)}", exc_info=True)
            return 0.5  # Return a neutral complexity if there's an error

    async def _get_topic_classification(self, text: str) -> tuple:
        """
        Retrieves topic classification and confidence score from LLM.
        """
        try:
            self.logger.debug("Requesting topic classification from LLM.")
            topic_result = await self._get_llm_response(
                prompt=[
                    {"role": "system", "content": "Classify the following text into one of these categories: politics, technology, entertainment, science. Respond with a JSON object containing 'topic' and 'confidence'."},
                    {"role": "user", "content": text}
                ],
                llm_call_type="topic_classification"
            )
            if not topic_result:
                raise ValueError("Empty response from LLM for topic classification.")

            # Attempt to parse JSON response
            try:
                topic_data = json.loads(topic_result)
                topic = topic_data.get('topic', 'unknown').strip().lower()
                topic_focus = float(topic_data.get('confidence', 0.0))
            except (json.JSONDecodeError, TypeError, ValueError) as parse_error:
                self.logger.error(f"Invalid topic classification format: '{topic_result}'. Error: {parse_error}")
                topic, topic_focus = "unknown", 0.0

            self.logger.debug(f"Topic: {topic}, Topic focus: {topic_focus}")
            return topic, topic_focus

        except Exception as e:
            self.logger.error(f"Error in _get_topic_classification: {str(e)}", exc_info=True)
            return "unknown", 0.0

    async def _get_sentiment_analysis(self, text: str) -> float:
        """
        Retrieves sentiment analysis and computes emotional valence.
        """
        try:
            self.logger.debug("Requesting sentiment analysis from LLM.")
            sentiment_result = await self._get_llm_response(
                prompt=[
                    {"role": "system", "content": "Analyze the sentiment of the following text. Respond with a JSON object containing 'sentiment' ('POSITIVE' or 'NEGATIVE') and 'confidence' (0-1)."},
                    {"role": "user", "content": text}
                ],
                llm_call_type="sentiment_analysis"
            )
            if not sentiment_result:
                raise ValueError("Empty response from LLM for sentiment analysis.")

            # Attempt to parse JSON response
            try:
                sentiment_data = json.loads(sentiment_result)
                sentiment = sentiment_data.get('sentiment', 'NEUTRAL').strip().upper()
                score = float(sentiment_data.get('confidence', 0.0))
                emotional_valence = score if sentiment == 'POSITIVE' else -score
            except (json.JSONDecodeError, TypeError, ValueError) as parse_error:
                self.logger.error(f"Invalid sentiment analysis format: '{sentiment_result}'. Error: {parse_error}")
                emotional_valence = 0.0

            self.logger.debug(f"Emotional valence: {emotional_valence}")
            return emotional_valence

        except Exception as e:
            self.logger.error(f"Error in _get_sentiment_analysis: {str(e)}", exc_info=True)
            return 0.0

    def _get_additional_metrics(self, text: str) -> tuple:
        """
        Computes arousal and dominance using TextBlob or other methods.
        """
        try:
            self.logger.debug("Computing additional metrics (arousal and dominance).")
            if self.default_analysis_method == 'textblob':
                blob = TextBlob(text)
                arousal = abs(blob.sentiment.polarity)  # Simplistic measure
                dominance = blob.sentiment.subjectivity  # Simplistic measure
            else:
                # Placeholder for other analysis methods
                arousal = 0.5
                dominance = 0.5

            self.logger.debug(f"Arousal: {arousal}, Dominance: {dominance}")
            return arousal, dominance

        except Exception as e:
            self.logger.error(f"Error in _get_additional_metrics: {str(e)}", exc_info=True)
            return 0.0, 0.0

    def _compute_attention_allocation(self, text: str) -> float:
        """
        Computes attention allocation based on word count and average word length.
        """
        try:
            self.logger.debug("Computing attention allocation.")
            words = text.split()
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            attention_allocation = min(1.0, (len(words) / 100) * (avg_word_length / 5))
            self.logger.debug(f"Attention allocation: {attention_allocation}")
            return attention_allocation
        except Exception as e:
            self.logger.error(f"Error in _compute_attention_allocation: {str(e)}", exc_info=True)
            return 0.0

    async def _get_memory_activation(self, text: str) -> float:
        """
        Retrieves memory activation estimation from LLM.
        """
        try:
            self.logger.debug("Requesting memory activation estimation from LLM.")
            memory_activation_result = await self._get_llm_response(
                prompt=[
                    {"role": "system", "content": "Estimate the memory activation for the following text. Respond with a JSON object containing 'memory_activation' as a float between 0 and 1."},
                    {"role": "user", "content": text}
                ],
                llm_call_type="memory_activation"
            )
            if not memory_activation_result:
                raise ValueError("Empty response from LLM for memory activation.")

            # Attempt to parse JSON response
            try:
                memory_data = json.loads(memory_activation_result)
                memory_activation = float(memory_data.get('memory_activation', 0.5))
                memory_activation = max(0.0, min(memory_activation, 1.0))  # Clamp between 0 and 1
            except (json.JSONDecodeError, TypeError, ValueError) as parse_error:
                self.logger.error(f"Invalid memory activation format: '{memory_activation_result}'. Error: {parse_error}")
                memory_activation = 0.5  # Default value

            self.logger.debug(f"Memory activation: {memory_activation}")
            return memory_activation

        except Exception as e:
            self.logger.error(f"Error in _get_memory_activation: {str(e)}", exc_info=True)
            return 0.5  # Default value on error

    def _compute_cognitive_metrics(self, text: str, dominance: float, topic_focus: float, attention_allocation: float) -> tuple:
        """
        Computes cognitive control, consciousness level, cognitive load, and mental energy.
        """
        try:
            self.logger.debug("Computing cognitive metrics.")
            cognitive_control = 1 - min(1.0, abs(TextBlob(text).sentiment.subjectivity - 0.5) * 2)
            consciousness_level = (topic_focus + attention_allocation + cognitive_control) / 3
            words = text.split()
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            cognitive_load = min(1.0, (len(words) / 200) + (avg_word_length / 10))
            mental_energy = 1 - cognitive_load

            self.logger.debug(f"Cognitive control: {cognitive_control}, "
                              f"Consciousness level: {consciousness_level}, "
                              f"Cognitive load: {cognitive_load}, "
                              f"Mental energy: {mental_energy}")
            return cognitive_control, consciousness_level, cognitive_load, mental_energy
        except Exception as e:
            self.logger.error(f"Error in _compute_cognitive_metrics: {str(e)}", exc_info=True)
            return 0.0, 0.0, 0.0, 1.0  # Default values on error

    def _compute_creativity_metrics(self, text: str, topic_focus: float) -> tuple:
        """
        Computes curiosity level, uncertainty, and creativity index.
        """
        try:
            self.logger.debug("Computing creativity metrics.")
            words = text.split()
            unique_words = set(words)
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

            curiosity_level = len(unique_words) / len(words) if words else 0.0
            uncertainty = 1 - topic_focus
            creativity_index = len(unique_words) / (avg_word_length * 5) if avg_word_length > 0 else 0.0

            self.logger.debug(f"Curiosity level: {curiosity_level}, "
                              f"Uncertainty: {uncertainty}, "
                              f"Creativity index: {creativity_index}")
            return curiosity_level, uncertainty, creativity_index
        except Exception as e:
            self.logger.error(f"Error in _compute_creativity_metrics: {str(e)}", exc_info=True)
            return 0.0, 1.0, 0.0  # Default values on error

    def _assemble_measurement_vector(self, *args) -> np.ndarray:
        """
        Assembles all metrics into a single measurement vector of length 108.
        """
        try:
            self.logger.debug("Assembling measurement vector.")
            measurement_vector = np.array(args)

            # Ensure the vector is of length 108 by padding with zeros if necessary
            if measurement_vector.size < 108:
                padding = np.zeros(108 - measurement_vector.size)
                measurement_vector = np.concatenate((measurement_vector, padding))
                self.logger.debug(f"Measurement vector padded with {108 - measurement_vector.size} zeros.")
            elif measurement_vector.size > 108:
                measurement_vector = measurement_vector[:108]
                self.logger.warning("Measurement vector truncated to 108 elements.")

            self.logger.debug(f"Measurement vector shape: {measurement_vector.shape}")
            return measurement_vector

        except Exception as e:
            self.logger.error(f"Error in _assemble_measurement_vector: {str(e)}", exc_info=True)
            return np.zeros(108)

    async def _get_llm_response(self, prompt: list, llm_call_type: str) -> str:
        """
        Helper method to get responses from the LLM, handling both async generators and direct responses.

        Args:
            prompt (list): The list of messages to send to the LLM.
            llm_call_type (str): The type of LLM call for logging purposes.

        Returns:
            str: The concatenated response from the LLM.
        """
        response = ""
        try:
            self.logger.debug(f"Sending prompt to LLM for {llm_call_type}: {prompt}")
            response_generator = await self.provider_manager.generate_response(
                messages=prompt,
                llm_call_type=llm_call_type
            )
            if hasattr(response_generator, "__aiter__"):  # Check if it's an async generator
                async for chunk in response_generator:
                    response += chunk
            else:
                response = response_generator
            response = response.strip()
            self.logger.debug(f"LLM response for {llm_call_type}: {response}")
            return response
        except Exception as e:
            self.logger.error(f"Error in _get_llm_response ({llm_call_type}): {str(e)}", exc_info=True)
            return ""
