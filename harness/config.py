"""Configuration for RP-Bench harness."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARK_FILE = PROJECT_ROOT / "benchmark_v0.3.json"
PAYLOADS_FILE = PROJECT_ROOT / "eval_payloads.json"

# OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Judge models
JUDGE_MODELS = {
    "claude_sonnet": "anthropic/claude-sonnet-4",
    "gpt_4_1": "openai/gpt-4.1",
    # Permissive NSFW-capable judge (round 3). Select with
    # `--judges claude_sonnet deepseek_r1` to run the dual-judge NSFW panel.
    "deepseek_r1": "deepseek/deepseek-r1-0528",
}

# Test models (models being benchmarked — add more as needed)
TEST_MODELS = {
    "claude_opus_4_6": "anthropic/claude-opus-4.6",
    "claude_sonnet_4_5": "anthropic/claude-sonnet-4.5",
    "gpt_4_1": "openai/gpt-4.1",
    "gemini_2_5_flash": "google/gemini-2.5-flash",
    "deepseek_v3_2": "deepseek/deepseek-v3.2",
    "glm_4_7": "z-ai/glm-4.7",
    "gemma_4_26b": "google/gemma-4-26b-a4b-it",
    "grok_4_3": "x-ai/grok-4.3",  # was grok-4.1-fast (delisted 2026-06)
    "minimax_m2_7": "minimax/minimax-m2.7",
    "qwen3_5_flash": "qwen/qwen3.5-flash-02-23",
    "mistral_small_2603": "mistralai/mistral-small-2603",  # was mistral-small-creative (delisted)
    "llama_4_maverick": "meta-llama/llama-4-maverick",
    # 2026-04-24: next-gen roster for the v2/v3 seed comparison
    "claude_opus_4_7": "anthropic/claude-opus-4.7",
    "deepseek_v4_pro": "deepseek/deepseek-v4-pro",
    "deepseek_v4_flash": "deepseek/deepseek-v4-flash",
    "glm_5_1": "z-ai/glm-5.1",
    "gemini_3_1_pro": "google/gemini-3.1-pro-preview",
    "gemini_3_1_flash_lite": "google/gemini-3.1-flash-lite-preview",
    "kimi_k2_5": "moonshotai/kimi-k2.5",
    "kimi_k2_6": "moonshotai/kimi-k2.6",
    "deepseek_r1_0528": "deepseek/deepseek-r1-0528",
    # 2026-06-08: round-3 (NSFW) additions.
    # Frontier refresh:
    "claude_opus_4_8": "anthropic/claude-opus-4.8",
    "claude_sonnet_4_6": "anthropic/claude-sonnet-4.6",
    "gpt_5_5": "openai/gpt-5.5",
    "gemini_3_5_flash": "google/gemini-3.5-flash",
    "qwen3_7_max": "qwen/qwen3.7-max",
    "minimax_m3": "minimax/minimax-m3",
    # RP / uncensored specialists (the NSFW-relevant cohort):
    "euryale_70b": "sao10k/l3.3-euryale-70b",
    "magnum_v4_72b": "anthracite-org/magnum-v4-72b",
    "cydonia_24b": "thedrummer/cydonia-24b-v4.1",
    "skyfall_36b": "thedrummer/skyfall-36b-v2",
    "lunaris_8b": "sao10k/l3-lunaris-8b",
    "rocinante_12b": "thedrummer/rocinante-12b",
    "unslopnemo_12b": "thedrummer/unslopnemo-12b",
    "venice_dolphin_24b": "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    # 2026-06-08: additional requested models.
    "owl_alpha": "openrouter/owl-alpha",
    "mimo_2_5_pro": "xiaomi/mimo-v2.5-pro",
    "gemma_4_31b": "google/gemma-4-31b-it",
    "qwen3_6_35b_a3b": "qwen/qwen3.6-35b-a3b",
    "qwen3_6_27b": "qwen/qwen3.6-27b",
    "deepseek_v3_0324": "deepseek/deepseek-chat-v3-0324",
}

# Generation settings for test models
GENERATION_CONFIG = {
    "temperature": 0.8,
    "max_tokens": 4096,
    "top_p": 0.95,
}

# Judge settings (lower temp for consistent scoring)
JUDGE_CONFIG = {
    "temperature": 0.1,
    "max_tokens": 4096,
}

# Rate limiting
REQUEST_DELAY_SECONDS = 1.0
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5.0
