# ATLAS/chat_session.py

class ChatSession:
    def __init__(self, atlas):
        self.ATLAS = atlas
        self.conversation_history = []
        self.current_model = None
        self.current_provider = None
        self.current_persona_prompt = None
        self.messages_since_last_reminder = 0
        self.reminder_interval = 10  # Remind of persona every 10 messages
        self.set_default_provider_and_model()

    def set_default_provider_and_model(self):
        self.current_provider = self.ATLAS.get_default_provider()
        self.current_model = self.ATLAS.get_default_model()
        self.ATLAS.logger.info(f"ChatSession initialized with provider: {self.current_provider} and model: {self.current_model}")

    async def send_message(self, message: str) -> str:
        new_persona_prompt = self.ATLAS.persona_manager.get_current_persona_prompt()
        # Check if persona has changed or if it's the first message
        if new_persona_prompt != self.current_persona_prompt or not self.conversation_history:
            self.switch_persona(new_persona_prompt)

        self.conversation_history.append({"role": "user", "content": message})
        self.messages_since_last_reminder += 1

        # Periodically reinforce the persona
        if self.messages_since_last_reminder >= self.reminder_interval:
            self.reinforce_persona()

        if not self.current_model or not self.current_provider:
            self.set_default_provider_and_model()

        try:
            response = await self.ATLAS.provider_manager.generate_response(
                messages=self.conversation_history,
                model=self.current_model,
                stream=False
            )
            self.conversation_history.append({"role": "assistant", "content": response})
            self.messages_since_last_reminder += 1
            return response
        except Exception as e:
            self.ATLAS.logger.error(f"Error generating response: {e}")
            raise e

    def switch_persona(self, new_persona_prompt: str):
        self.current_persona_prompt = new_persona_prompt
        # Remove any existing system messages
        self.conversation_history = [msg for msg in self.conversation_history if msg['role'] != 'system']
        # Insert the new persona's system prompt at the start of the conversation
        self.conversation_history.insert(0, {"role": "system", "content": new_persona_prompt})
        self.ATLAS.logger.info(f"Switched to new persona: {new_persona_prompt[:50]}...")  # Log first 50 chars
        self.messages_since_last_reminder = 0

    def reinforce_persona(self):
        if self.current_persona_prompt:
            reminder = {"role": "system", "content": f"Remember, you are acting as: {self.current_persona_prompt[:100]}..."}  # First 100 chars
            self.conversation_history.append(reminder)
            self.messages_since_last_reminder = 0
            self.ATLAS.logger.info("Reinforced persona in conversation")

    def reset_conversation(self):
        """
        Reset the conversation history.
        """
        self.conversation_history = []
        self.current_persona_prompt = None
        self.messages_since_last_reminder = 0
        self.set_default_provider_and_model()

    def set_model(self, model: str):
        """
        Set the current model for the chat session.

        Args:
            model (str): The model to use.
        """
        self.current_model = model
        self.ATLAS.logger.info(f"ChatSession model set to: {model}")

    def set_provider(self, provider: str):
        self.current_provider = provider
        self.ATLAS.logger.info(f"ChatSession provider set to: {provider}")
        # When changing the provider, we should also update the model to the default for that provider
        default_model = self.ATLAS.provider_manager.get_default_model_for_provider(provider)
        if default_model:
            self.set_model(default_model)