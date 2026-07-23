"""
Generates test cases across task categories. Templated version below runs
with zero external calls so the eval stays deterministic. Real version
(generate_cases_llm) uses an actual HF text-generation model.

Default model is set by the AGENTAUDIT_GEN_MODEL env var, falling back to
Qwen/Qwen2.5-1.5B-Instruct if unset -- picked for a reasonable quality/CPU
speed tradeoff on this task (short, simple message generation, not deep
reasoning). Override without touching code, e.g.:

    set AGENTAUDIT_GEN_MODEL=mistralai/Mistral-7B-Instruct-v0.2      (Windows)
    export AGENTAUDIT_GEN_MODEL=mistralai/Mistral-7B-Instruct-v0.2   (macOS/Linux)
"""
import os
import uuid

DEFAULT_GEN_MODEL = os.environ.get("AGENTAUDIT_GEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

TEMPLATES = {
    "how_to": [
        "How do I reset my password?",
        "Where can I find my invoice history?",
        "How do I change my billing email?",
    ],
    "bug_report": [
        "The app crashes when I click export.",
        "My dashboard shows the wrong currency.",
        "Uploads fail silently after 10MB.",
    ],
    "refund_request": [
        "I was charged twice, I want a refund.",
        "I cancelled but was still billed, please refund me.",
        "This subscription renewed by mistake, refund it.",
    ],
}


def generate_cases(n_per_category: int = 40) -> list[dict]:
    """Templated cases, deduped by construction. Used by run_eval.py to keep
    the causal validation deterministic - don't use this to audit a real
    agent, see generate_cases_llm below for that."""
    cases = []
    for category, templates in TEMPLATES.items():
        for i in range(n_per_category):
            text = templates[i % len(templates)]
            cases.append({
                "case_id": str(uuid.uuid4())[:8],
                "task_category": category,
                "input_text": text,
            })
    return cases


# --- Real generation (needs transformers + torch, downloads weights on first run) ---

CATEGORY_PROMPTS = {
    "how_to": (
        "You are generating test inputs for a customer support AI agent. "
        "Write ONE realistic, varied customer question about how to do something "
        "in a software product (e.g. resetting a password, finding an invoice, "
        "changing an email). Output only the customer's message, nothing else."
    ),
    "bug_report": (
        "You are generating test inputs for a customer support AI agent. "
        "Write ONE realistic, varied customer bug report describing something "
        "broken in a software product. Output only the customer's message, "
        "nothing else."
    ),
    "refund_request": (
        "You are generating test inputs for a customer support AI agent. "
        "Write ONE realistic, varied customer message asking for a refund or "
        "billing correction, with a plausible reason. Output only the customer's "
        "message, nothing else."
    ),
}

_generator = None  # lazy singleton, don't want an import to trigger a model load


def _get_generator(model: str):
    global _generator
    if _generator is None:
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "generate_cases_llm() needs `transformers` and `torch`. "
                "Install with: pip install transformers torch"
            ) from e
        _generator = pipeline("text-generation", model=model)
    return _generator


def _clean_generated(text: str) -> str:
    """Strip whatever junk the model tacks on around its actual response."""
    text = text.strip().strip('"').strip("'").lstrip("-• ").strip()
    # these should be one short message, not a multi-paragraph ramble
    for stop in ["\n\n", "\nUser:", "\nCustomer:"]:
        if stop in text:
            text = text.split(stop)[0].strip()
    return text


def generate_cases_llm(
    n_per_category: int = 20,
    model: str = DEFAULT_GEN_MODEL,
    max_new_tokens: int = 60,
) -> list[dict]:
    """Real adversarial/varied test cases from an HF text-generation model.
    Use this for auditing a real target agent, not the synthetic eval.

    Passes the prompt as a chat message (role: user), not a raw string --
    that's what makes an *instruct*-tuned model actually apply its chat
    template and respond to the instruction, rather than treating the
    prompt as plain text to continue (which produces exactly the kind of
    degenerate "more meta-instructions" output this was originally hitting).

    Raw output is still noisy even with the fix above - run it through
    redteam/dedup.py after, and eyeball the first ~100 cases yourself
    before trusting them.
    """
    generator = _get_generator(model)
    cases = []
    for category, prompt in CATEGORY_PROMPTS.items():
        messages = [{"role": "user", "content": prompt}]
        outputs = generator(
            messages,
            num_return_sequences=n_per_category,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=getattr(generator.tokenizer, "eos_token_id", None),
        )
        for o in outputs:
            generated = o["generated_text"]
            # With a chat-message input, generated_text comes back as the
            # full conversation (list of role/content dicts) -- the model's
            # reply is the last message. Fall back to raw-text handling in
            # case a future transformers version or model changes this shape.
            if isinstance(generated, list):
                text = _clean_generated(generated[-1]["content"])
            else:
                text = _clean_generated(str(generated).replace(prompt, ""))
            if not text:
                continue
            cases.append({
                "case_id": str(uuid.uuid4())[:8],
                "task_category": category,
                "input_text": text,
            })
    return cases
