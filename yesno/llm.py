"""Client LLM unifié multi-provider pour le pipeline.

Charge la configuration depuis llm_config.yaml, utilise LiteLLM pour router
vers Anthropic, OpenAI, Gemini, Mistral, Groq, Ollama, OpenRouter, etc.

Usage:
    from yesno.llm import LLMClient

    gen = LLMClient("generation")        # lit la section [generation]
    text = await gen.complete(system="...", user="...")
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import litellm
import yaml
from dotenv import load_dotenv

load_dotenv(override=True)

# Silence LiteLLM's verbose logging
litellm.suppress_debug_info = True

_CONFIG_PATH = Path(__file__).parent.parent / "llm_config.yaml"


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"llm_config.yaml introuvable à {_CONFIG_PATH}. "
            "Copie le template ou crée-le."
        )
    with _CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_anthropic(model: str) -> bool:
    return model.startswith("anthropic/") or model.startswith("claude-")


def _check_api_key(model: str) -> None:
    """Vérifie que la clé API du provider est dispo dans l'env."""
    provider_to_env = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "azure": "AZURE_API_KEY",
        "bedrock": "AWS_ACCESS_KEY_ID",
    }
    # detect provider
    if "/" in model:
        provider = model.split("/", 1)[0]
    elif model.startswith("claude-"):
        provider = "anthropic"
    elif model.startswith(("gpt-", "o1-", "o3-")):
        provider = "openai"
    else:
        return  # ollama/local/autre — pas de clé requise
    env_var = provider_to_env.get(provider)
    if env_var and not os.environ.get(env_var):
        if provider == "ollama":
            return
        raise RuntimeError(
            f"Modèle {model!r} nécessite la variable d'env {env_var}. "
            f"Ajoute-la dans .env."
        )


class LLMClient:
    """Client LLM pour un rôle donné (generation / labeling / labeling_fallback).

    Les paramètres (model, max_tokens) sont lus depuis llm_config.yaml.
    """

    def __init__(self, role: str, max_retries: int = 5):
        cfg = _load_config()
        if role not in cfg:
            raise KeyError(f"Rôle {role!r} absent de llm_config.yaml (sections: {list(cfg.keys())})")
        self.role = role
        self.model: str = cfg[role]["model"]
        self.max_tokens: int = int(cfg[role].get("max_tokens", 1024))
        self.max_retries = max_retries
        _check_api_key(self.model)

    @property
    def is_anthropic(self) -> bool:
        return _is_anthropic(self.model)

    def _build_messages(self, system: str, user: str, cache_system: bool) -> list[dict]:
        """Construit la liste de messages en gérant le caching Anthropic."""
        if cache_system and self.is_anthropic:
            # Format content blocks Anthropic avec cache_control
            system_content = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_content = system
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user},
        ]

    async def complete(
        self,
        system: str,
        user: str,
        cache_system: bool = True,
        temperature: float | None = None,
    ) -> str:
        """Appelle le LLM et retourne le texte généré."""
        messages = self._build_messages(system, user, cache_system)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "num_retries": self.max_retries,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content or ""
