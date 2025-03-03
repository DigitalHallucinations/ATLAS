# User Manual

## Getting Started with the CLI

1. Open a terminal and navigate to the chat application directory.
2. Activate the virtual environment:
   ```
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Start the CLI:
   ```
   python -m Modules.UI.cli
   ```

## Available Commands

The CLI presents a menu with the following options:

1. Set log level
2. Set provider
3. Set model
4. View current settings
5. HuggingFace Options (if applicable)
6. Chat settings
7. Start chat interface
8. Exit

### Setting Log Level

- Choose option 1 from the main menu.
- Enter the desired log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

### Setting Provider

- Choose option 2 from the main menu.
- Select a provider from the list of available options.

### Setting Model

- Choose option 3 from the main menu.
- Select a model from the list of available options for the current provider.

### Viewing Current Settings

- Choose option 4 from the main menu to display current configuration.

### HuggingFace Options

- Choose option 5 from the main menu (only available when HuggingFace is the current provider).
- This submenu allows you to manage HuggingFace models and settings.

### Chat Settings

- Choose option 6 from the main menu.
- Adjust settings like temperature, max tokens, etc.

### Starting Chat Interface

- Choose option 7 from the main menu to begin chatting.
- Type your messages and press Enter to send.
- Type 'exit' to end the chat session.
- Type 'change_system' to modify the system message.

## Managing Chat Sessions

- At the start of a chat interface, you'll be prompted to create a new session or load an existing one.
- To load an existing session, you'll need to provide the session ID.

## Customizing Chat Settings

In the Chat Settings menu, you can adjust:

- Streaming (on/off)
- Temperature (0.0 to 1.0)
- Max Tokens
- Top P (0.0 to 1.0)
- Frequency Penalty (-2.0 to 2.0)
- Presence Penalty (-2.0 to 2.0)

## Troubleshooting Common Issues

1. If the CLI fails to start, ensure your virtual environment is activated and all dependencies are installed.
2. If you encounter API errors, verify your API keys in the `.env` file.
3. For database-related issues, check your MongoDB connection and ensure the service is running.

For more detailed troubleshooting, refer to the Troubleshooting and FAQ document.