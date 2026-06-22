"""llm_models — per-model rate-limit configurations (Groq Free Plan).

Usage::

    from llm_models import get_model_config, MODEL_REGISTRY

    cfg = get_model_config("llama-3.3-70b-versatile")
    print(cfg.requests_per_minute, cfg.tokens_per_minute)       # 30  12000

    # Direct class access (respects env-var overrides)
    from llm_models import Llama3370BVersatileConfig
    cfg = Llama3370BVersatileConfig.from_env()

    # Iterate every known model
    for model_id, cls in MODEL_REGISTRY.items():
        print(model_id, cls().active_limits())
"""
from __future__ import annotations

from exceptions.request import UnknownModelError
from llm_models.base import ModelRateLimitConfig
from llm_models.allam_2_7b import Allam27BConfig
from llm_models.canopylabs_orpheus_arabic_saudi import OrpheusArabicSaudiConfig
from llm_models.canopylabs_orpheus_v1_english import OrpheusV1EnglishConfig
from llm_models.groq_compound import CompoundConfig
from llm_models.groq_compound_mini import CompoundMiniConfig
from llm_models.llama_3_1_8b_instant import Llama318BInstantConfig
from llm_models.llama_3_3_70b_versatile import Llama3370BVersatileConfig
from llm_models.meta_llama_4_scout import Llama4Scout17BConfig
from llm_models.meta_llama_prompt_guard_22m import LlamaPromptGuard222MConfig
from llm_models.meta_llama_prompt_guard_86m import LlamaPromptGuard286MConfig
from llm_models.openai_gpt_oss_120b import GptOss120BConfig
from llm_models.openai_gpt_oss_20b import GptOss20BConfig
from llm_models.openai_gpt_oss_safeguard_20b import GptOssSafeguard20BConfig
from llm_models.qwen3_32b import Qwen332BConfig
from llm_models.qwen3_6_27b import Qwen3627BConfig
from llm_models.whisper_large_v3 import WhisperLargeV3Config
from llm_models.whisper_large_v3_turbo import WhisperLargeV3TurboConfig


# ---------------------------------------------------------------------------
# Registry — Groq model ID → config class
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type[ModelRateLimitConfig]] = {
	"allam-2-7b": Allam27BConfig,
	"canopylabs/orpheus-arabic-saudi": OrpheusArabicSaudiConfig,
	"canopylabs/orpheus-v1-english": OrpheusV1EnglishConfig,
	"groq/compound": CompoundConfig,
	"groq/compound-mini": CompoundMiniConfig,
	"llama-3.1-8b-instant": Llama318BInstantConfig,
	"llama-3.3-70b-versatile": Llama3370BVersatileConfig,
	"meta-llama/llama-4-scout-17b-16e-instruct": Llama4Scout17BConfig,
	"meta-llama/llama-prompt-guard-2-22m": LlamaPromptGuard222MConfig,
	"meta-llama/llama-prompt-guard-2-86m": LlamaPromptGuard286MConfig,
	"openai/gpt-oss-120b": GptOss120BConfig,
	"openai/gpt-oss-20b": GptOss20BConfig,
	"openai/gpt-oss-safeguard-20b": GptOssSafeguard20BConfig,
	"qwen/qwen3-32b": Qwen332BConfig,
	"qwen/qwen3.6-27b": Qwen3627BConfig,
	"whisper-large-v3": WhisperLargeV3Config,
	"whisper-large-v3-turbo": WhisperLargeV3TurboConfig,
	}


def get_model_config(model_id: str) -> ModelRateLimitConfig:
	"""Return an env-initialised :class:`ModelRateLimitConfig` for *model_id*.

	Calls ``SubClass.from_env()`` so any ``<ENV_PREFIX>_<FIELD>`` env vars
	are applied on top of the hardcoded Free Plan defaults.

	Args:
		model_id: Groq API model identifier (e.g. ``"llama-3.3-70b-versatile"``).

	Returns:
		Initialised :class:`ModelRateLimitConfig` subclass instance.

	Raises:
		UnknownModelError: if *model_id* is not registered.
	"""
	if model_id not in MODEL_REGISTRY:
		available = sorted(MODEL_REGISTRY)
		raise UnknownModelError(
			f"Unknown model {model_id!r}. Registered llm_models: {available}",
			model_id=model_id,
			available_models=available,
			)
	return MODEL_REGISTRY[model_id].from_env()


__all__: list = [
	"ModelRateLimitConfig",
	"Allam27BConfig",
	"OrpheusArabicSaudiConfig",
	"OrpheusV1EnglishConfig",
	"CompoundConfig",
	"CompoundMiniConfig",
	"Llama318BInstantConfig",
	"Llama3370BVersatileConfig",
	"Llama4Scout17BConfig",
	"LlamaPromptGuard222MConfig",
	"LlamaPromptGuard286MConfig",
	"GptOss120BConfig",
	"GptOss20BConfig",
	"GptOssSafeguard20BConfig",
	"Qwen332BConfig",
	"Qwen3627BConfig",
	"WhisperLargeV3Config",
	"WhisperLargeV3TurboConfig",
	"MODEL_REGISTRY",
	"get_model_config",
	]
