# Security Warning

**IMPORTANT: Immediate Action Required**

This deployment bundle may contain sensitive API keys or credentials configured in `.env` or passed via environment variables.

1.  **Rotate Keys**: Upon deployment, immediately rotate the `DEEPSEEK_API_KEY` used in this environment.
2.  **Access Control**: Ensure the `.env` file is readable only by the application owner (`chmod 600 .env`).
3.  **Logs**: Monitor logs to ensure `Authorization` headers or raw keys are not being printed. The current code has been patched to avoid this, but vigilance is required.
4.  **Repository**: Do NOT commit `.env` or any file containing real keys to a public repository.
