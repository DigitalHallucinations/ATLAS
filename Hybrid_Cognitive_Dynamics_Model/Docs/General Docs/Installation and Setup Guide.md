# Installation and Setup Guide

## System Requirements

- Python 3.8 or higher
- MongoDB 4.4 or higher
- 8GB RAM (minimum), 16GB RAM (recommended)
- 50GB free disk space for model storage

## Installation Process

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/chat-application.git
   cd chat-application
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file in the root directory with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   MISTRAL_API_KEY=your_mistral_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   GOOGLE_API_KEY=your_google_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   MONGO_CONNECTION_STRING=your_mongodb_connection_string
   LOG_LEVEL=INFO
   DEFAULT_PROVIDER=OpenAI
   DEFAULT_MODEL=gpt-4
   REAL_AGENT_ROOT=/path/to/real/agent/directory
   ```

2. Replace the placeholder values with your actual API keys and settings.

## Database Setup

1. Install MongoDB on your system if not already installed.

2. Start the MongoDB service:
   ```
   sudo systemctl start mongodb
   ```

3. Create a new database for the chat application:
   ```
   mongo
   > use chat_application_db
   > db.createCollection("chat_sessions")
   > db.createCollection("messages")
   > exit
   ```

4. Update the `MONGO_CONNECTION_STRING` in your `.env` file with the appropriate connection string.

## Verifying Installation

1. Run the following command to start the CLI:
   ```
   python -m Modules.UI.cli
   ```

2. If the CLI starts without errors, your installation is successful.

## Troubleshooting

- If you encounter any "Module not found" errors, ensure you've activated the virtual environment and installed all dependencies.
- If you have database connection issues, check your MongoDB service status and the connection string in the `.env` file.
- For API-related errors, verify that you've entered the correct API keys in the `.env` file.

For more detailed troubleshooting, refer to the Troubleshooting and FAQ document.