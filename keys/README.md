# API Credentials & Key Rotation Guide

1. **Configuration**: Copy `api_keys.example.json` to `api_keys.json` and supply your Gemini API key(s) or OpenAI API key.
2. **Automatic 429 Failover**: When using `GEMINI_API_KEYS` array, any key hitting an HTTP 429 rate limit or quota ceiling automatically rotates to the next available standby key without restarting the run.
3. **Fallback Resolution**: If `api_keys.json` is missing, the pipeline gracefully resolves credentials from `.env` or system environment variables.
4. **Security**: `keys/api_keys.json` is strictly git-ignored in `.gitignore` to guarantee zero secret leaks in version control.
