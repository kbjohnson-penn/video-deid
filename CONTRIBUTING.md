# Contributing to Video-DeID

First off, thank you for considering contributing to Video-DeID! It's people like you that make this tool better for everyone.

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report. Following these guidelines helps maintainers understand your report, reproduce the behavior, and find related reports.

- Use the bug report template
- Use a clear and descriptive title
- Describe the exact steps which reproduce the problem
- Provide specific examples to demonstrate the steps
- Describe the behavior you observed after following the steps
- Explain which behavior you expected to see instead and why
- Include screenshots or video clips if possible
- Include details about your environment

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion, including completely new features and minor improvements to existing functionality.

- Use the feature request template
- Use a clear and descriptive title
- Provide a step-by-step description of the suggested enhancement
- Provide specific examples to demonstrate the steps or point out related functionality
- Describe the current behavior and explain which behavior you expected to see instead
- Explain why this enhancement would be useful to most users

### Pull Requests

- Fill in the required template
- Do not include issue numbers in the PR title
- Follow the Python style guide
- Include tests for any new functionality
- Ensure all tests pass
- Update documentation for any changed functionality

## Development Setup

1. Fork the Video-DeID repo
2. Clone your fork locally
3. Create a new branch for your feature or fix
4. Install development dependencies:
   ```
   pip install -e ".[dev]"
   ```
5. Make your changes
6. Write tests for your changes
7. Run tests:
   ```
   pytest
   ```
8. Update documentation if needed
9. Commit your changes (see [Commit Message Guidelines](#commit-message-guidelines))
10. Push your branch to your fork
11. Submit a pull request

## Commit Message Guidelines

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line
- Consider using the following pattern:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `style:` for formatting changes
  - `refactor:` for code refactoring
  - `test:` for adding tests
  - `chore:` for maintenance tasks

## Code Style

- Follow PEP 8 guidelines
- Use descriptive variable names
- Add docstrings to all functions, classes, and modules
- Keep functions small and focused
- Use type hints where appropriate

## Documentation

- Update the README.md with details of changes to the interface
- Update the docs directory for more complex features
- Add or update examples if needed

## License

By contributing, you agree that your contributions will be licensed under the project's license.
