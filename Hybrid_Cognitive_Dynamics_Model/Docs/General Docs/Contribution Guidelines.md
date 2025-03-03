# Contribution Guidelines

We welcome contributions to the chat application project! This document outlines the process for contributing and the standards we adhere to.

## How to Contribute

1. **Fork the Repository**: Start by forking the main repository to your own GitHub account.

2. **Clone the Fork**: Clone your fork to your local machine:
   ```
   git clone https://github.com/your-username/chat-application.git
   ```

3. **Create a Branch**: Create a new branch for your feature or bug fix:
   ```
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes**: Implement your changes, following the coding standards outlined below.

5. **Test Your Changes**: Ensure all tests pass and add new tests for new functionality.

6. **Commit Your Changes**: Make concise, descriptive commits:
   ```
   git commit -m "Add feature: brief description of the feature"
   ```

7. **Push to Your Fork**: Push your changes to your GitHub fork:
   ```
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request**: Go to the original repository on GitHub and create a pull request from your branch.

## Coding Standards

We follow PEP 8 style guide for Python code. Additionally:

1. Use meaningful variable and function names.
2. Write docstrings for all functions, classes, and modules.
3. Use type hints to improve code readability and catch potential type-related errors.
4. Keep functions focused and small, adhering to the Single Responsibility Principle.
5. Comment complex logic, but prefer self-explanatory code where possible.
6. Use f-strings for string formatting.

Example of good code style:

```python
from typing import List, Dict

def process_messages(messages: List[Dict[str, str]]) -> str:
    """
    Process a list of message dictionaries and return a formatted string.

    Args:
        messages (List[Dict[str, str]]): A list of message dictionaries.

    Returns:
        str: A formatted string representing the processed messages.
    """
    formatted_messages = []
    for message in messages:
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        formatted_messages.append(f"{role.capitalize()}: {content}")
    
    return "\n".join(formatted_messages)
```

## Pull Request Process

1. Ensure your code adheres to the coding standards.
2. Update the README.md with details of changes, if applicable.
3. Add or update tests as necessary.
4. Update the documentation to reflect any changes.
5. Ensure all tests pass before submitting the pull request.
6. The pull request will be reviewed by maintainers. Be open to feedback and be prepared to make changes.
7. Once approved, a maintainer will merge your pull request.

## Reporting Issues

- Use the GitHub issue tracker to report bugs.
- Before creating a new issue, please check if it has already been reported.
- Provide a clear and descriptive title for the issue.
- Describe the exact steps to reproduce the problem.
- Explain the behavior you expected to see.
- Include any relevant code snippets or error messages.

## Feature Requests

We're always looking for suggestions to improve our project. To submit a feature request:

1. Check if the feature has already been suggested or implemented.
2. Provide a clear and detailed explanation of the feature.
3. Explain why this feature would be useful to most users.
4. Consider writing the feature yourself and contributing it via a pull request.

## Code of Conduct

### Our Pledge

We pledge to make participation in our project and community a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behavior include:

- The use of sexualized language or imagery and unwelcome sexual attention or advances
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information, such as a physical or electronic address, without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team. All complaints will be reviewed and investigated and will result in a response that is deemed necessary and appropriate to the circumstances. The project team is obligated to maintain confidentiality with regard to the reporter of an incident.

## Questions?

If you have any questions about contributing, please feel free to contact the project maintainers.

Thank you for your interest in contributing to the chat application project! We appreciate your effort to help make this project better.