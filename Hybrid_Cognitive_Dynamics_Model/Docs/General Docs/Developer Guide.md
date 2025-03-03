# Developer Guide

## Code Structure and Organization

The chat application is organized into several key modules:

```
chat-application/
├── modules/
│   ├── Providers/
│   │   ├── OpenAI/
│   │   ├── Mistral/
│   │   ├── Google/
│   │   ├── HuggingFace/
│   │   ├── Anthropic/
│   │   ├── model_manager.py
│   │   └── provider_manager.py
│   ├── UI/
│   │   ├── base_cli.py
│   │   ├── cli.py
│   │   ├── chat_settings_manager.py
│   │   └── HF_manager.py
│   ├── config.py
│   ├── Chat_Bot.py
│   └── chat.py
├── Database/
│   └── DB_manager.py
├── requirements.txt
└── .env
```

## Adding New AI Providers

1. Create a new directory under `modules/Providers/` for your provider.
2. Implement a generator class similar to existing providers (e.g., `OA_gen_response.py`).
3. Update `provider_manager.py` to include the new provider:
   - Add the provider to the `provider_config` dictionary.
   - Implement any provider-specific logic in the `switch_llm_provider` method.
4. Update `model_manager.py` to load models for the new provider.
5. Add the new provider's API key to the `.env` file and `config.py`.

## Extending CLI Functionality

1. To add new CLI commands, extend the `CLI` class in `cli.py`.
2. Add a new method for your command.
3. Update the `display_menu` method to include your new option.
4. Implement the logic for handling the new option in the `handle_choice` method.

## Best Practices for Contributing

1. Follow PEP 8 style guide for Python code.
2. Write docstrings for all classes and methods.
3. Use type hints to improve code readability and catch potential type-related errors.
4. Write unit tests for new functionality using pytest.
5. Keep methods focused and small, adhering to the Single Responsibility Principle.
6. Use meaningful variable and function names.
7. Comment complex logic, but prefer self-explanatory code where possible.

## Testing Procedures

1. Write unit tests for individual components in a `tests/` directory.
2. Use pytest for running tests:
   ```
   pytest tests/
   ```
3. Implement integration tests to ensure different components work together correctly.
4. For UI testing, consider using tools like `pytest-mock` to simulate user inputs.
5. Aim for high test coverage, especially for critical paths in the application.

## Debugging Tips

1. Use the Python debugger (pdb) or an IDE like PyCharm for step-by-step debugging.
2. Leverage logging throughout the application for easier troubleshooting.
3. For async code, use `asyncio.run(debug=True)` to get more detailed error information.

## Performance Optimization

1. Use caching mechanisms where appropriate, especially for frequently accessed data.
2. Optimize database queries by ensuring proper indexing and using aggregation pipelines where possible.
3. For CPU-intensive tasks, consider using multiprocessing to leverage multiple cores.
4. Profile the application using tools like cProfile to identify performance bottlenecks.

## Version Control Guidelines

1. Use feature branches for developing new features or fixing bugs.
2. Write clear, concise commit messages describing the changes made.
3. Squash commits before merging to maintain a clean history.
4. Use pull requests for code reviews before merging into the main branch.

By following these guidelines, you'll be able to contribute effectively to the chat application project while maintaining code quality and consistency.