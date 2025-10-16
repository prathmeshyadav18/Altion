import os
from dotenv import load_dotenv
try:
    from langchain_groq import ChatGroq  # Groq provider
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "ChatGroq not found. Install with: pip install langchain-groq"
    ) from e
try:
    from langchain_openai import ChatOpenAI  # OpenAI/OpenRouter-compatible provider
except Exception:
    ChatOpenAI = None  # Will error with guidance if user selects openrouter


def get_groq_api_key() -> str:
    """Load env file and return GROQ API key or raise helpful error."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env or environment."
        )
    return api_key


def get_llm(model_name: str, provider: str = "groq"):
    """Create and return a configured chat model client for the chosen provider.

    Providers:
    - groq: uses langchain_groq.ChatGroq with `GROQ_API_KEY`
    - openrouter: uses langchain_openai.ChatOpenAI with `OPENAI_API_KEY` and `OPENAI_BASE_URL=https://openrouter.ai/api/v1`
    """
    provider = provider.lower().strip()
    if provider == "groq":
        groq_api_key = get_groq_api_key()
        return ChatGroq(
            temperature=0.2,
            model_name=model_name,
            api_key=groq_api_key,
                            )
    elif provider == "openrouter":
        if ChatOpenAI is None:
            raise ImportError(
                "langchain-openai is required for OpenRouter. Install with: pip install langchain-openai"
            )
        base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. For OpenRouter, use your OpenRouter key."
            )
        return ChatOpenAI(
            temperature=0.2,
            model=model_name,
            api_key=api_key,
            base_url=base_url,
        )
    else:
        raise ValueError("Unsupported provider. Use 'groq' or 'openrouter'.")


def _invoke_once(model: str, prompt: str, provider: str) -> str:
    llm = get_llm(model, provider)
    resp = llm.invoke(prompt)
    return getattr(resp, "content", resp)


def simple_demo(prompt: str | None = None) -> None:
    """Run a simple call to verify the Groq client works."""
    prompt = prompt or "Reply with a short, friendly greeting."

    provider = os.getenv("PROVIDER", "groq").lower()
    # If user provided explicit model env, use that for the selected provider.
    configured_model = (
        os.getenv("GROQ_MODEL") if provider == "groq" else os.getenv("OPENROUTER_MODEL") or os.getenv("MODEL")
    )
    if configured_model:
        try:
            content = _invoke_once(configured_model, prompt, provider)
            print(content)
            return
        except Exception as e:
            msg = str(e)
            if provider == "groq" and (
                "model_decommissioned" in msg or "has been decommissioned" in msg
            ):
                print(
                    "Groq model is deprecated. Update GROQ_MODEL in .env to a supported model."
                )
            raise

    if provider == "openrouter":
        # For OpenRouter, require the user to set an explicit model like 'openai/gpt-oss-20b'
        raise RuntimeError(
            "Set OPENROUTER_MODEL or MODEL in .env to a supported OpenRouter model, e.g., openai/gpt-oss-20b."
        )
    else:
        # Otherwise, try a few common Groq model IDs until one works.
        candidates = [
            # Older stable IDs
            "llama3-8b-8192",
            "llama3-70b-8192",
            # Newer preview/stable variants that may be available
            "llama-3.2-11b-text-preview",
            "llama-3.2-90b-text-preview",
            "llama-3.1-8b-instant",
        ]
        last_err: Exception | None = None
        for m in candidates:
            try:
                content = _invoke_once(m, prompt, provider)
                print(f"[model: {m}]\n{content}")
                return
            except Exception as e:
                last_err = e
                msg = str(e)
                if not ("model_decommissioned" in msg or "has been decommissioned" in msg):
                    # For non-decommission errors (e.g., invalid key), stop early.
                    break
                continue

        # If we got here, nothing worked. Provide guidance.
        print(
            "No working Groq model tried. Set GROQ_MODEL in .env to a supported model from your Groq console."
        )
        if last_err:
            raise last_err
        else:
            raise RuntimeError("Unable to contact Groq or invalid configuration.")


if __name__ == "__main__":
    # Minimal smoke test with optional CLI prompt
    import argparse

    parser = argparse.ArgumentParser(description="Simple Groq LLM demo")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to send")
    args = parser.parse_args()

    simple_demo(args.prompt)
