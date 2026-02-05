# Contributing to ConvNeXt Earthquake Precursor Detection

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/convnext-earthquake-precursor.git`
3. Create a virtual environment: `python -m venv venv`
4. Install dependencies: `pip install -r requirements.txt`
5. Install development dependencies: `pip install pytest black flake8`

## Development Workflow

### Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **Type hints** where appropriate

Before submitting:
```bash
black src/ scripts/ tests/
flake8 src/ scripts/ tests/
```

### Running Tests

```bash
pytest tests/ -v
```

### Making Changes

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Add tests for new functionality
4. Run tests and linting
5. Commit with clear messages: `git commit -m "Add feature: description"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Open a Pull Request

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Include tests for new features
- Update documentation if needed
- Ensure all tests pass

## Reporting Issues

When reporting issues, please include:
- Python version
- PyTorch version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages (if any)

## Areas for Contribution

- **Model improvements**: New architectures, training techniques
- **Data augmentation**: New augmentation strategies
- **Visualization**: Better GradCAM, feature visualization
- **Documentation**: Tutorials, examples, API docs
- **Testing**: More comprehensive test coverage
- **Performance**: Optimization, inference speed

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Questions?

Open an issue or contact the maintainers.

Thank you for contributing!
