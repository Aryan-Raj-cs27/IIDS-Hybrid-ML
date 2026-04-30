# Contributing to IIDS

Thank you for your interest in contributing to the Intelligent Intrusion Detection System! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Report issues professionally
- Respect intellectual property

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/IIDS.git`
3. Create a virtual environment: `python -m venv .venv`
4. Activate it: `.\.venv\Scripts\activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Create a feature branch: `git checkout -b feature/your-feature-name`

## Development Guidelines

### Python Code Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints: `def predict(data: np.ndarray) -> str:`
- Write docstrings for all functions and classes
- Use meaningful variable names

### Commit Messages
```
Prefix: Short description (50 chars max)

Optional detailed explanation of changes.
- Point 1
- Point 2

Fixes #issue_number
```

**Common Prefixes**:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation update
- `style:` Code style (formatting, etc.)
- `refactor:` Code restructuring
- `perf:` Performance improvement
- `test:` Test additions/updates
- `security:` Security enhancement

### Example
```
feat: Add LIME explainability for model predictions

Implements LIME integration for interpreting individual
model decisions in frontend modal.
- Add lime dependency to requirements.txt
- Create explanation generator in backend
- Update threat modal with explainability section

Fixes #42
```

## Testing

Before submitting a PR:
```bash
# Run linting
flake8 backend/ model/

# Run type checking (if installed)
mypy backend/

# Manual testing
python backend/app.py
# Test all endpoints in Postman/curl
```

## Pull Request Process

1. **Update your branch**
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Push your changes**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Open Pull Request**
   - Use clear title: "Add LIME explainability feature"
   - Link related issues: "Fixes #42"
   - Describe changes and testing performed
   - Request 1-2 reviewers

4. **Address feedback**
   - Make requested changes
   - Push updates (same branch)
   - PR automatically updates

5. **Merge**
   - Maintainers merge after approval
   - Delete feature branch after merge

## Areas for Contribution

### High Priority
- [ ] Production authentication (OAuth 2.0 / JWT)
- [ ] HTTPS/TLS implementation
- [ ] Rate limiting and DDoS protection
- [ ] Database encryption
- [ ] Comprehensive unit tests
- [ ] CI/CD pipeline (GitHub Actions)

### Medium Priority
- [ ] Docker containerization
- [ ] Kubernetes deployment configs
- [ ] Model explainability (LIME/SHAP)
- [ ] Elasticsearch integration for logs
- [ ] REST API versioning
- [ ] GraphQL endpoint

### Community-Driven
- [ ] Performance benchmarks
- [ ] Additional datasets support
- [ ] Mobile app companion
- [ ] Visualization improvements
- [ ] Documentation translations
- [ ] Tutorial videos

## Reporting Issues

### Security Issues
**DO NOT** open public issues for security vulnerabilities!
Email maintainers privately with:
- Vulnerability description
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

### Bug Reports
Include:
```markdown
**Environment**
- OS: Windows/macOS/Linux
- Python version: 3.13.x
- TensorFlow version: 2.15.x

**Description**
[Clear description of the bug]

**Steps to Reproduce**
1. ...
2. ...
3. ...

**Expected Behavior**
[What should happen]

**Actual Behavior**
[What actually happened]

**Error Message/Logs**
[Full error trace]

**Screenshots**
[If applicable]
```

### Feature Requests
Include:
```markdown
**Is this related to a problem?**
[Describe the use case]

**Proposed Solution**
[Your idea]

**Alternatives Considered**
[Other approaches]

**Additional Context**
[References, images, etc.]
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings for code changes
- Update API docs if endpoints change
- Create/update diagrams if architecture changes

## Review Process

- Maintainers aim to review within 3-5 business days
- Constructive feedback provided for improvements
- Small fixes in separate commits vs. squashing large PRs
- At least one approval before merge

## Resources

- [Python PEP 8](https://www.python.org/dev/peps/pep-0008/)
- [Git Workflow](https://git-scm.com/book/en/v2/Git-Branching-Branching-Workflow)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Semantic Versioning](https://semver.org/)

## Questions?

- Check existing issues/discussions first
- Open a discussion for questions
- Tag maintainers if urgent

---

**Thank you for making IIDS better! 🚀**
