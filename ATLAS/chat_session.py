# ATLAS/chat_session.py

from typing import List, Dict

class ChatSession:
    """
    Manages a chat session, including conversation history and interactions
    with the language model via the ProviderManager.
    """

    def __init__(self, atlas):
        """
        Initialize the ChatSession with an empty conversation history.

        Args:
            atlas (ATLAS): The main ATLAS instance.
        """
        self.ATLAS = atlas
        self.conversation_history = []  # List of messages
        self.current_model = None

    async def send_message(self, message: str) -> str:
        """
        Send a message to the model and get the response.

        Args:
            message (str): The user's message.

        Returns:
            str: The model's response.
        """
        # Add user's message to the conversation history
        self.conversation_history.append({"role": "user", "content": message})

        # Ensure the current model is set
        if not self.current_model:
            self.current_model = self.ATLAS.provider_manager.get_current_model()
            if not self.current_model:
                raise ValueError("No model selected")

        # Generate response using the ProviderManager
        try:
            # Add 'await' here
            response = await self.ATLAS.provider_manager.generate_response(
                messages=self.conversation_history,
                model=self.current_model,
                stream=False
            )
            # Add the model's response to the conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        except Exception as e:
            self.ATLAS.logger.error(f"Error generating response: {e}")
            raise e
    
    def reset_conversation(self):
        """
        Reset the conversation history.
        """
        self.conversation_history = []

    def set_model(self, model: str):
        """
        Set the current model for the chat session.

        Args:
            model (str): The model to use.
        """
        self.current_model = model
        # Optionally, set the model in the provider manager
        # self.ATLAS.provider_manager.set_model(model)
