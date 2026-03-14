# Contributing to Agent DevTools

First off, thank you for considering contributing to Agent DevTools! It's people like you that make this tool better for everyone debugging AI agents.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## Getting Started

### Good First Issues

Looking for a place to start? Check out issues labeled [`good first issue`](https://github.com/Sigmabrogz/agent-devtools/labels/good%20first%20issue) - these are specifically curated for new contributors.

### Types of Contributions We're Looking For

- **Bug fixes** - Found something broken? We'd love a fix!
- **New instrumentors** - Support for more LLM providers (Cohere, Mistral, Groq, etc.)
- **UI improvements** - Make the debugging experience even better
- **Documentation** - Tutorials, examples, better explanations
- **Performance** - Make tracing even more lightweight
- **Tests** - Increase coverage and reliability

## Development Setup

### Prerequisites

- Python 3.9+
- Git

### Installation

```bash
# Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/agent-devtools.git
cd agent-devtools

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agentdbg --cov-report=html

# Run specific test file
pytest tests/unit/test_core.py

# Run integration tests
pytest tests/integration/
```

### Running Linters

```bash
# Check code style
ruff check src tests

# Type checking
mypy src

# Format code
black src tests
```

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates.

When filing a bug, include:

1. **Python version** (`python --version`)
2. **Package version** (`pip show agent-devtools`)
3. **OS and version**
4. **Steps to reproduce**
5. **Expected vs actual behavior**
6. **Relevant logs/screenshots**

### Suggesting Features

We love feature suggestions! Please:

1. Check if it's already been suggested
2. Explain the use case clearly
3. Describe the expected behavior
4. Consider if it fits the project's scope

### Your First Code Contribution

1. **Find an issue** - Look for `good first issue` or `help wanted` labels
2. **Comment on the issue** - Let us know you're working on it
3. **Fork and branch** - Create a branch from `main`
4. **Make changes** - Write code, tests, and docs
5. **Submit PR** - Reference the issue in your PR

## Pull Request Process

### Before Submitting

- [ ] Code follows the style guide (run `ruff check` and `black`)
- [ ] All tests pass (`pytest`)
- [ ] Type hints are added for new code (`mypy src`)
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and descriptive

### PR Title Format

Use conventional commits format:

- `feat: add Cohere instrumentor`
- `fix: handle empty response in OpenAI wrapper`
- `docs: add tutorial for custom breakpoints`
- `test: add integration tests for LangChain`
- `refactor: simplify span creation logic`

### PR Description Template

```markdown
## Summary
Brief description of changes

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Related Issues
Fixes #123
```

### Review Process

1. A maintainer will review your PR
2. Address any feedback
3. Once approved, a maintainer will merge

## Style Guide

### Python Style

- Follow PEP 8
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use `ruff` for linting and `black` for formatting

### Code Organization

```
src/agentdbg/
├── __init__.py      # Public API exports
├── cli.py           # CLI commands
├── config.py        # Configuration classes
├── core.py          # Core tracing logic
├── models.py        # Data models (Span, Trace, etc.)
├── instrumentors/   # Auto-instrumentation for LLM providers
├── server/          # WebSocket server
├── storage/         # Trace persistence
└── ui/              # Web UI assets
```

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Keep first line under 72 characters
- Reference issues when relevant

### Documentation

- Use docstrings for all public functions/classes
- Include type hints in docstrings
- Add examples for complex functionality

## Adding a New Instrumentor

Want to add support for a new LLM provider? Here's how:

1. Create `src/agentdbg/instrumentors/your_provider_instrumentor.py`
2. Implement the `Instrumentor` interface
3. Add to `auto_instrument()` in `__init__.py`
4. Add tests in `tests/unit/test_instrumentors.py`
5. Add example in `examples/`
6. Update README with the new provider

Example structure:

```python
from agentdbg.instrumentors.base import BaseInstrumentor

class YourProviderInstrumentor(BaseInstrumentor):
    def instrument(self):
        # Wrap the provider's API calls
        pass
    
    def uninstrument(self):
        # Remove wrapping
        pass
```

## Community

- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - Questions and ideas
- **Twitter** - Follow [@Sigmabrogz](https://twitter.com/Sigmabrogz) for updates

## Recognition

Contributors are recognized in:
- The README contributors section
- Release notes for significant contributions
- Our eternal gratitude for making agent debugging less painful!

---

Thank you for contributing! Every bug fix, feature, and documentation improvement helps developers debug their AI agents more effectively.
