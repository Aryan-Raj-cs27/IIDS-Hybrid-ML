# Security Policy

## Reporting Security Vulnerabilities

**IMPORTANT: DO NOT report security vulnerabilities via public GitHub issues!**

If you discover a security vulnerability in IIDS, please email the maintainers privately:

📧 **Email**: [Contact maintainer privately through GitHub]

Include in your report:
- Vulnerability description
- Affected component(s)
- Steps to reproduce
- Potential impact assessment
- Suggested remediation (if any)
- Your name and contact information (optional)

We will acknowledge receipt within 48 hours and provide updates every 5 business days.

## Security Best Practices

### For Users

1. **Authentication**
   - Change default credentials immediately
   - Implement production-grade authentication (OAuth 2.0, JWT)
   - Use strong, unique passwords
   - Enable MFA if available

2. **Network Security**
   - Run IIDS behind a reverse proxy (nginx, Apache)
   - Use HTTPS/TLS for all connections
   - Restrict API access by IP whitelist if possible
   - Use VPN for remote access

3. **Data Protection**
   - Encrypt sensitive data at rest
   - Use parameterized queries (already implemented)
   - Regularly backup forensic database
   - Implement proper access controls

4. **Monitoring**
   - Enable comprehensive logging
   - Monitor for unauthorized access
   - Set up alerts for anomalies
   - Perform regular security audits

### For Developers

1. **Code Security**
   - No hardcoded secrets
   - Use environment variables for configuration
   - Validate all user input
   - Sanitize database queries (implemented)
   - Handle errors gracefully without exposing internals

2. **Dependencies**
   ```bash
   # Regular security updates
   pip install --upgrade pip
   pip install -U -r requirements.txt
   pip check  # Check for known vulnerabilities
   ```

3. **Version Management**
   - Keep track of dependencies in requirements.txt
   - Use specific versions: `TensorFlow==2.15.0`
   - Avoid using wildcard versions: `TensorFlow==2.*`

4. **Testing**
   - Write security-focused tests
   - Test input validation thoroughly
   - Use static analysis tools (Bandit)

5. **Deployment**
   ```bash
   # Never use debug mode in production
   FLASK_ENV=production
   FLASK_DEBUG=false
   
   # Use production WSGI server
   # pip install gunicorn
   # gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app
   ```

## Vulnerability Disclosure Timeline

- **Day 1**: Vulnerability report received
- **Day 1-2**: Acknowledgment and initial assessment
- **Day 3-7**: Fix development and testing
- **Day 7-14**: Security patch release
- **Day 15**: Public disclosure (after patch availability)

## Known Security Considerations

### Authentication (Development Only)
The current demo authentication (hardcoded credentials) is **for development/testing only**. 
Production deployments MUST implement:
- OAuth 2.0 / OpenID Connect
- JWT tokens with proper expiration
- Session management with secure cookies
- Rate limiting on login attempts

### Database
- SQLite is suitable for development/small deployments
- Production systems should use PostgreSQL or MySQL with encryption
- Database connections should use SSL/TLS
- Regular backups must be encrypted and stored securely

### API Security
- Implement rate limiting to prevent brute force
- Use CORS properly with specific origins
- Validate Content-Type headers
- Implement request signing for sensitive operations
- Use API keys with scope restrictions

### Data Privacy
- Comply with GDPR, CCPA, and local regulations
- Implement data retention policies
- Allow user data export and deletion
- Encrypt sensitive data in transit and at rest

## Supported Versions

| Version | Status | Security Updates |
|---------|--------|------------------|
| 1.0+ | Current | Yes |
| < 1.0 | Beta | Best effort |

## Security Checklist for Deployments

- [ ] Change all default credentials
- [ ] Enable HTTPS/TLS
- [ ] Configure CORS properly
- [ ] Set up rate limiting
- [ ] Implement authentication system
- [ ] Encrypt database
- [ ] Enable access logging
- [ ] Configure firewall rules
- [ ] Set up monitoring/alerts
- [ ] Regular security audits
- [ ] Keep dependencies updated
- [ ] Implement backup/disaster recovery
- [ ] Test incident response plan
- [ ] Document security architecture

## Third-Party Security Tools

We recommend using:
- **Bandit**: Python security issue scanner
  ```bash
  pip install bandit
  bandit -r backend/ model/
  ```

- **Safety**: Dependency vulnerability checker
  ```bash
  pip install safety
  safety check
  ```

- **OWASP Dependency-Check**: Comprehensive dependency scanner

## Questions?

For security-related questions (non-vulnerability):
- Open a discussion on GitHub
- Tag security-related topics

---

**Last Updated**: April 30, 2026 | **Status**: Active
