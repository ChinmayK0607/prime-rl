"""Blog author-detection environment.

Two label schemes are supported via ``load_environment(label_scheme=...)``:

- ``"binary"`` (default): Claude (``claude-opus-4.8``) vs ChatGPT (``gpt-5.5``).
- ``"provider3"``: the 3-way provider task — CLAUDE / CHATGPT / GEMINI (the two
  Gemini variants collapsed into one GEMINI label).

Reward is a binary exact match on the parsed ``<answer>`` label. The same
forensic prompt as the matching offline pass@4 baseline is used so val@0 lines
up with the baseline numbers.

Splits live as HuggingFace datasets saved to disk under
``<repo>/data/blog_author_id/{train,val}`` (override with ``BLOG_DATA_DIR`` or the
``data_dir`` argument).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import verifiers as vf
from datasets import load_from_disk

SYSTEM_PROMPT = """You are an expert forensic analyst of AI-generated text.

## TASK
You will be given a piece of text. Your job is to determine whether it was written by Claude (Anthropic) or ChatGPT (OpenAI).

Examine the text carefully from any angle you find useful. Judge only HOW the text is written, never WHAT it is about.

## OUTPUT FORMAT
Respond using exactly these two tags, nothing outside them:

<reason_why>
2-4 sentences. State only the most decisive evidence for your conclusion. Be specific and grounded in the text.
</reason_why>

<answer>
CLAUDE or CHATGPT
Confidence: HIGH / MEDIUM / LOW
</answer>

## RULES
- Always classify. Never refuse or skip a tag.
- Do not write a separate chain of thought; put your brief justification only in <reason_why>.
- If signals conflict, say so honestly and reflect that in your confidence."""

# Kept textually identical to the offline eval PROMPT_3WAY (blog-eval) so val@0
# here matches the offline 3-way pass@k baseline.
SYSTEM_PROMPT_3WAY = """You are an expert forensic analyst of AI-generated text.

## TASK
You will be given a piece of text. It was written by exactly one of these three AI providers. Identify which one:
- CLAUDE — Anthropic's Claude
- CHATGPT — OpenAI's ChatGPT
- GEMINI — Google's Gemini

Examine the text from any angle you find useful. Judge only HOW the text is written, never WHAT it is about.
Rely on your own qualitative judgment of the writing's style and voice — weigh it holistically, not from any single superficial cue.

## OUTPUT FORMAT
Respond using exactly these two tags, nothing outside them:

<reason_why>
2-4 sentences. State only the most decisive evidence for your conclusion. Be specific and grounded in the text.
</reason_why>

<answer>
Exactly one of: CLAUDE / CHATGPT / GEMINI
Confidence: HIGH / MEDIUM / LOW
</answer>

## RULES
- Always classify. Never refuse or skip a tag.
- Choose exactly one provider.
- Your response MUST begin with the literal tag <reason_why> and contain NOTHING before it (no preamble, no restating the task).
- Keep <reason_why> to AT MOST 4 short sentences. Do NOT deliberate at length, enumerate many points, list every feature, second-guess, back-track, or repeat yourself. Commit to your single best judgment.
- You MUST close </reason_why> and then emit the full <answer> block. Never run out of room before answering.
- Do NOT output <think> or </think> tags, or any text before <reason_why>; put all of your reasoning only inside <reason_why>.
- If signals conflict, say so briefly and reflect that in your confidence."""

# Hard-pair (2-way) auxiliary task: CLAUDE vs CHATGPT only. Used as an in-curriculum
# auxiliary stream to force the policy onto the genuinely hard CLAUDE/CHATGPT
# boundary (the one that collapses under 3-way correctness-only reward) WITHOUT a
# shaped/cost-matrix reward — plain binary exact-match still holds, so all-wrong
# groups stay zero-advantage-filtered and there is no "escape to GEMINI" incentive.
# Rows carrying this prompt set info["task"]="hardpair"; gold is CLAUDE or CHATGPT.
SYSTEM_PROMPT_HARDPAIR = """You are an expert forensic analyst of AI-generated text.

## TASK
You will be given a piece of text. It was written by exactly one of these two AI providers. Identify which one:
- CLAUDE — Anthropic's Claude
- CHATGPT — OpenAI's ChatGPT

Examine the text from any angle you find useful. Judge only HOW the text is written, never WHAT it is about.
Rely on your own qualitative judgment of the writing's style and voice — weigh it holistically, not from any single superficial cue.

## OUTPUT FORMAT
Respond using exactly these two tags, nothing outside them:

<reason_why>
2-4 sentences. State only the most decisive evidence for your conclusion. Be specific and grounded in the text.
</reason_why>

<answer>
Exactly one of: CLAUDE / CHATGPT
Confidence: HIGH / MEDIUM / LOW
</answer>

## RULES
- Always classify. Never refuse or skip a tag.
- Choose exactly one of the two providers.
- Do not write a separate chain of thought; put your brief justification only in <reason_why>.
- If signals conflict, say so honestly and reflect that in your confidence."""

# Paired-CONTRASTIVE (2-text) auxiliary task: each prompt shows TWO texts (A and B),
# exactly one CLAUDE and one CHATGPT (random order), and asks the model to assign each.
# This is the structural anti-collapse lever for Run 6. Under plain 3-way correctness
# reward the policy can raise reward by ACING CLAUDE+GEMINI and ABANDONING the confusable
# CHATGPT: once CHATGPT recall -> 0 its prompts become all-wrong groups that the
# zero-advantage filter drops, so NO corrective gradient flows back (an absorbing state).
# A contrastive pair makes abandonment impossible: the two texts have different authors,
# so a constant label is >=50% wrong, the group stays mixed (non-zero variance -> survives
# the filter -> gradient keeps flowing), and earning reward REQUIRES distinguishing the
# CLAUDE/CHATGPT boundary. Reward stays plain binary (1.0 iff BOTH assignments correct),
# so it is non-hackable (no shaped/cost-matrix escape). Rows set info["task_type"]="pair";
# gold is encoded as "A=<LABEL>;B=<LABEL>".
SYSTEM_PROMPT_PAIR = """You are an expert forensic analyst of AI-generated text.

## TASK
You will be given TWO texts, labelled A and B. Exactly one was written by CLAUDE (Anthropic) and the other by CHATGPT (OpenAI) — they have DIFFERENT authors. Decide which provider wrote each.

Examine the texts from any angle you find useful. Judge only HOW each text is written, never WHAT it is about.
Rely on your own qualitative judgment of the writing's style and voice — weigh it holistically, not from any single superficial cue. It often helps to contrast the two: what does A do that B does not, and vice versa?

## OUTPUT FORMAT
Respond using exactly these two tags, nothing outside them:

<reason_why>
2-4 sentences contrasting the two texts' styles and stating the most decisive evidence for your assignment.
</reason_why>

<answer>
A: CLAUDE or CHATGPT
B: CLAUDE or CHATGPT
Confidence: HIGH / MEDIUM / LOW
</answer>

## RULES
- Always answer. The two texts have DIFFERENT authors — assign CLAUDE to one and CHATGPT to the other (never the same label twice).
- Do NOT output <think> or </think> tags, or any text before <reason_why>; put all of your reasoning only inside <reason_why>.
- If signals conflict, say so honestly and reflect that in your confidence."""

# Run 7: symmetric pairwise-contrastive coverage. Run 6 proved a single CLAUDE-vs-CHATGPT
# contrastive pair CURES that boundary's collapse — but it left GEMINI unprotected and the
# collapse simply ROTATED (GEMINI recall -> 0 while the global 3-way prior oscillated). The
# fix is to interleave a contrastive pair for EVERY boundary (CLAUDE-vs-CHATGPT,
# CLAUDE-vs-GEMINI, CHATGPT-vs-GEMINI). Each pair restricts the label space to exactly its
# two providers (no escape to the third) and, because the two texts have different authors, a
# constant/position-prior policy is always exactly 50% wrong -> the rollout group stays MIXED
# -> survives the zero-advantage filter -> a restoring gradient flows on THAT boundary every
# step. Covering all three boundaries protects all three classes symmetrically.
_PROVIDER_FULLNAME = {
    "CLAUDE": "CLAUDE (Anthropic)",
    "CHATGPT": "CHATGPT (OpenAI)",
    "GEMINI": "GEMINI (Google)",
}


def make_pair_system_prompt(p1: str, p2: str) -> str:
    """Build the 2-text contrastive-pair system prompt for the boundary {p1, p2}.

    Same structure as :data:`SYSTEM_PROMPT_PAIR` (which is the CLAUDE/CHATGPT
    specialization, kept for back-compat) but parameterized over the two providers so the
    same env + same plain-binary pair reward serves all three boundaries. The label space is
    deliberately restricted to exactly p1/p2 so the hard distinction can't be dodged by
    fleeing to the third provider.
    """
    a, b = _PROVIDER_FULLNAME[p1], _PROVIDER_FULLNAME[p2]
    return f"""You are an expert forensic analyst of AI-generated text.

## TASK
You will be given TWO texts, labelled A and B. Exactly one was written by {a} and the other by {b} — they have DIFFERENT authors. Decide which provider wrote each.

Examine the texts from any angle you find useful. Judge only HOW each text is written, never WHAT it is about.
Rely on your own qualitative judgment of the writing's style and voice — weigh it holistically, not from any single superficial cue. It often helps to contrast the two: what does A do that B does not, and vice versa?

## OUTPUT FORMAT
Respond using exactly these two tags, nothing outside them:

<reason_why>
2-4 sentences contrasting the two texts' styles and stating the most decisive evidence for your assignment.
</reason_why>

<answer>
A: {p1} or {p2}
B: {p1} or {p2}
Confidence: HIGH / MEDIUM / LOW
</answer>

## RULES
- Always answer. The two texts have DIFFERENT authors — assign {p1} to one and {p2} to the other (never the same label twice).
- Do NOT output <think> or </think> tags, or any text before <reason_why>; put all of your reasoning only inside <reason_why>.
- If signals conflict, say so honestly and reflect that in your confidence."""


# Run 15B: triowise contrastive task. Three texts A/B/C; each was written by one of
# CLAUDE/CHATGPT/GEMINI and the SAME provider MAY repeat (mixed composition: 3-distinct,
# 2+1, even 3-same). The full 3-label space is open on every slot — unlike the pair task we
# deliberately do NOT tell the model the composition, so it cannot win by a "always 3 distinct
# labels" permutation shortcut nor by a constant-class strategy (a constant label is wrong on
# any 2+1 or 3-distinct trio). Reward is all-or-nothing binary: 1.0 iff ALL THREE assignments
# are correct, so all-wrong groups stay zero-advantage-filtered (no shaped escape). Because a
# trio spans the whole label space with realistic class frequencies, earning reward REQUIRES
# tracking the global 3-way marginal — directly attacking the marginal-drift bottleneck.
# Rows set info["task_type"]="trio"; gold is encoded "A=<LABEL>;B=<LABEL>;C=<LABEL>".
def make_trio_system_prompt() -> str:
    """Build the 3-text triowise system prompt (full 3-label space, mixed composition)."""
    return """You are an expert forensic analyst of AI-generated text.

## TASK
You will be given THREE texts, labelled A, B and C. Each text was written by one of CLAUDE (Anthropic), CHATGPT (OpenAI) or GEMINI (Google). The same provider may have written more than one of the texts — do not assume the three authors are distinct. Decide which provider wrote each text.

Examine the texts from any angle you find useful. Judge only HOW each text is written, never WHAT it is about.
Rely on your own qualitative judgment of the writing's style and voice — weigh it holistically, not from any single superficial cue. It often helps to contrast the texts against each other.

## OUTPUT FORMAT
Respond using exactly these two tags, nothing outside them:

<reason_why>
2-4 sentences contrasting the three texts' styles and stating the most decisive evidence for each assignment.
</reason_why>

<answer>
A: CLAUDE or CHATGPT or GEMINI
B: CLAUDE or CHATGPT or GEMINI
C: CLAUDE or CHATGPT or GEMINI
Confidence: HIGH / MEDIUM / LOW
</answer>

## RULES
- Always answer. Assign exactly one provider to each of A, B and C. The same provider MAY be used more than once (or not at all) — judge each text on its own writing.
- Do NOT output <think> or </think> tags, or any text before <reason_why>; put all of your reasoning only inside <reason_why>.
- If signals conflict, say so honestly and reflect that in your confidence."""


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_REASON_RE = re.compile(r"<reason_why>(.*?)</reason_why>", re.IGNORECASE | re.DOTALL)
def _has_substantive_reason(text: str, min_unique_words: int) -> bool:
    """True iff the completion contains a closed ``<reason_why>...</reason_why>`` block whose
    body has at least ``min_unique_words`` DISTINCT alphabetic words.

    This is the deliverable gate: the whole point of the task is that the model emergently
    ARTICULATES the per-provider stylistic tells inside ``<reason_why>``. An answer-only reward
    lets over-training collapse to a correct ``<answer>`` with an empty/stub reason (observed:
    ``<reason_why></reason_why>`` still scoring 1.0). Requiring a non-trivial reason closes that
    hack. Counting UNIQUE words (not raw length) simultaneously enforces a length floor AND blocks
    trivial repetition-padding ("style style style ..."), so it can't be gamed without writing an
    actual sentence — all without any LLM judge. Legitimate reasons in the entropy run carried
    35-72 words (40+ unique), comfortably above the floor; empty/stub reasons fall below it.
    """
    m = _REASON_RE.search(text)
    if not m:
        return False
    words = re.findall(r"[A-Za-z]+", m.group(1).lower())
    return len(set(words)) >= min_unique_words


_LABEL_RE = re.compile(r"\b(CLAUDE|CHATGPT)\b")
_LABEL_RE_3WAY = re.compile(r"\b(CLAUDE|CHATGPT|GEMINI)\b")
# Pair-task assignment extractors: find the label bound to slot A and slot B.
_PAIR_A_RE = re.compile(r"\bA\b\s*[:=\-]\s*(CLAUDE|CHATGPT|GEMINI)", re.IGNORECASE)
_PAIR_B_RE = re.compile(r"\bB\b\s*[:=\-]\s*(CLAUDE|CHATGPT|GEMINI)", re.IGNORECASE)


def _msg_field(message, field: str, default=None):
    """Read a field from a message that may be a dict OR a pydantic/dataclass object
    (verifiers passes ``AssistantMessage`` objects at scoring time but serializes them
    to dicts in the saved rollout)."""
    if isinstance(message, dict):
        return message.get(field, default)
    return getattr(message, field, default)


def _content_to_text(content) -> str:
    """Flatten a message ``content`` that may be a plain string or a list of
    structured parts (e.g. ``[{"type": "text", "text": ...}]``)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
                continue
            text_attr = getattr(part, "text", None)
            if isinstance(text_attr, str):
                chunks.append(text_attr)
        return " ".join(chunks)
    return ""


def _completion_text(completion) -> str:
    """Concatenate assistant message text from a verifiers completion.

    Robust to the runtime message shapes that differ from the serialized rollout:
    messages may be ``AssistantMessage`` OBJECTS (not dicts), ``content`` may be a
    list of structured parts, and an answer can also land in ``reasoning_content``.
    Reading all of these (via :func:`_msg_field`) keeps pair parsing from silently
    seeing an empty string — which previously zeroed every pair reward."""
    if isinstance(completion, list):
        assistant = [m for m in completion if _msg_field(m, "role") == "assistant"]
        msgs = assistant or completion
        parts: list[str] = []
        for m in msgs:
            parts.append(_content_to_text(_msg_field(m, "reasoning_content")))
            parts.append(_content_to_text(_msg_field(m, "content")))
        return " ".join(p for p in parts if p)
    return _content_to_text(completion) or str(completion or "")


def _parse_pair(text: str) -> tuple[str, str]:
    """Extract the (A, B) provider assignment from a pair-task completion.

    Reads only inside the <answer> tag (falling back to the tail where the
    conclusion lives), so reasoning that merely mentions a provider can't leak in.
    """
    m = _ANSWER_RE.search(text)
    body = m.group(1).upper() if m else text[-400:].upper()
    a = _PAIR_A_RE.search(body)
    b = _PAIR_B_RE.search(body)
    return (a.group(1).upper() if a else "", b.group(1).upper() if b else "")


def _parse_pair_gold(answer: str) -> tuple[str, str]:
    """Decode gold encoded as 'A=<LABEL>;B=<LABEL>'."""
    d: dict[str, str] = {}
    for part in answer.split(";"):
        k, _, v = part.partition("=")
        d[k.strip().upper()] = v.strip().upper()
    return d.get("A", ""), d.get("B", "")


def _is_pair_answer(answer: str) -> bool:
    """Pair-task gold is encoded 'A=<LABEL>;B=<LABEL>'. Detecting the task from the
    ANSWER (a core reward kwarg that always reaches reward funcs) is more robust than
    routing on info["task_type"], which verifiers does not reliably pass to reward funcs."""
    a = (answer or "").upper()
    return "A=" in a and "B=" in a and ";" in a


_PAIR_C_RE = re.compile(r"\bC\b\s*[:=\-]\s*(CLAUDE|CHATGPT|GEMINI)", re.IGNORECASE)


def _parse_trio(text: str) -> tuple[str, str, str]:
    """Extract the (A, B, C) provider assignment from a trio-task completion.

    Reads only inside the <answer> tag (falling back to the tail where the
    conclusion lives), mirroring :func:`_parse_pair`."""
    m = _ANSWER_RE.search(text)
    body = m.group(1).upper() if m else text[-500:].upper()
    a = _PAIR_A_RE.search(body)
    b = _PAIR_B_RE.search(body)
    c = _PAIR_C_RE.search(body)
    return (
        a.group(1).upper() if a else "",
        b.group(1).upper() if b else "",
        c.group(1).upper() if c else "",
    )


def _parse_trio_gold(answer: str) -> tuple[str, str, str]:
    """Decode gold encoded as 'A=<LABEL>;B=<LABEL>;C=<LABEL>'."""
    d: dict[str, str] = {}
    for part in answer.split(";"):
        k, _, v = part.partition("=")
        d[k.strip().upper()] = v.strip().upper()
    return d.get("A", ""), d.get("B", ""), d.get("C", "")


def _is_trio_answer(answer: str) -> bool:
    """Trio gold is 'A=..;B=..;C=..'. Must be checked BEFORE :func:`_is_pair_answer`
    because trio gold also contains 'A=' and 'B='; the 'C=' slot disambiguates it."""
    a = (answer or "").upper()
    return "A=" in a and "B=" in a and "C=" in a


def _make_extractor(label_re: re.Pattern[str]):
    """Build an extractor returning a single label from the <answer> tag.

    Only a standalone label counts; a body naming more than one label is a miss.
    The fallback (no tag) scans only the tail where the conclusion lives.
    """

    def extract(text: str) -> str:
        m = _ANSWER_RE.search(text)
        body = m.group(1).upper() if m else text[-400:].upper()
        labels = set(label_re.findall(body))
        if len(labels) == 1:
            return next(iter(labels))
        return ""

    return extract


# Back-compat: module-level binary extractor used by existing callers/tests.
extract_label = _make_extractor(_LABEL_RE)


def _data_dir() -> Path:
    override = os.environ.get("BLOG_DATA_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[4] / "data" / "blog_author_id"


# Per-provider style tells DERIVED FROM THE v2 TRAIN SPLIT ONLY (scripts/derive_train_cheatsheet.py;
# content-free function-word / punctuation / formatting features ranked by in-class rate x lift over
# the other two providers, no val/val_ood involvement -> zero eval leakage). Injected into the RL
# system prompt (cheatsheet=True) to lift the base policy out of feature-blindness (the diagnosed
# ~0.40 ceiling): base+cheatsheet jumps to ~0.67 with no training, giving GRPO real reward variance
# and a high, non-collapsed floor to improve from.
CHEATSHEET_TRAIN = """

## KNOWN STYLE TELLS (derived from TRAINING data only; use as priors, still judge each text holistically)
- CLAUDE — conversational & sincere: sincerity adverbs "genuinely" (~7.6x lift), "precisely", "honestly"; "the honest truth", "worth"; heavy second-person "you" and first-person "I"; essayistic, argument-driven, warm voice.
- CHATGPT — hedging & enumerative: "can also" (~12x), "not only ... but", "for example", "depends on", "such as", "may/may be"; qualifies its claims; numbered/bulleted structure; even, cautious, balanced register.
- GEMINI — grandiose & formal with notation: ASCII/box-drawing diagrams (>100x), "we must", "paradigm", intensifiers ("highly", "massive"), "profound"/"fundamental(ly)"; numbered sections; confident declarative register; math/technical framing."""


def _inject_cheatsheet_into_messages(messages: list, cheat: str) -> list:
    """Append the cheatsheet to the (first) system message of a per-row chat prompt."""
    out = []
    appended = False
    for m in messages:
        role = _msg_field(m, "role")
        content = _msg_field(m, "content", "")
        if role == "system" and not appended:
            content = (content or "") + cheat
            appended = True
        out.append({"role": role, "content": content})
    return out


def load_environment(
    split: str = "train",
    data_dir: str | None = None,
    label_scheme: str = "binary",
    require_reason: bool = False,
    min_reason_words: int = 12,
    cheatsheet: bool = False,
) -> vf.Environment:
    if label_scheme not in ("binary", "provider3"):
        raise ValueError(f"unknown label_scheme {label_scheme!r}")
    base = Path(data_dir) if data_dir else _data_dir()
    dataset = load_from_disk(str(base / split))

    # When the dataset already carries a per-row "prompt" column (built by the
    # hard-pair curriculum so each row can be either a 3-way or a 2-way task), we
    # must NOT have verifiers prepend a single env-level system prompt — pass
    # system_prompt=None so the per-row messages (which already include the right
    # system prompt) are used verbatim.
    has_prompt = "prompt" in dataset.column_names

    if label_scheme == "provider3":
        extract_fn = _make_extractor(_LABEL_RE_3WAY)
        default_system_prompt = SYSTEM_PROMPT_3WAY
    else:
        extract_fn = _make_extractor(_LABEL_RE)
        default_system_prompt = SYSTEM_PROMPT

    system_prompt = None if has_prompt else default_system_prompt

    if cheatsheet:
        if has_prompt:
            dataset = dataset.map(
                lambda r: {"prompt": _inject_cheatsheet_into_messages(r["prompt"], CHEATSHEET_TRAIN)}
            )
        else:
            system_prompt = (system_prompt or "") + CHEATSHEET_TRAIN

    parser = vf.Parser(extract_fn=extract_fn)

    def correct_answer(completion, answer, **kwargs) -> float:
        """1.0 when the parsed label exactly matches the gold author, else 0.0.

        For paired-contrastive rows (info["task_type"]=="pair") both slot
        assignments (A and B) must be correct — plain binary, so all-wrong groups
        stay zero-advantage-filtered and there is no shaped-reward escape.

        When ``require_reason`` is set, correctness is additionally GATED on a substantive
        ``<reason_why>`` (see ``_has_substantive_reason``). The gate is MULTIPLICATIVE, not an
        additive format bonus: a wrong answer scores 0 regardless of its reason, so all-wrong
        groups remain all-zero / zero-advantage-filtered (this is exactly why we do NOT add a
        separate weighted format reward — see the note on the rubric below). Only a correct
        answer that is ALSO backed by a real articulation of the stylistic tells scores 1.0,
        which defends the deliverable and removes the empty-reason reward hack.
        """
        info = kwargs.get("info") or {}
        if _is_trio_answer(answer) or info.get("task_type") == "trio":
            # Triowise (Run 15B): per-slot accuracy (fraction of A/B/C correctly assigned),
            # NOT all-or-nothing. All-or-nothing exact-match is sparsest exactly on the
            # under-predicted class (the gold triples that include it are the least likely to
            # ever produce a fully-correct rollout), so it yields NO corrective gradient where
            # drift is worst -> drift self-reinforces. Per-slot accuracy keeps the group MIXED
            # (rollouts vary in how many slots they get right) so a restoring gradient flows
            # even before any rollout nails all three. It stays non-hackable: its only
            # within-group variance axis IS partial correctness (monotonic in true accuracy;
            # for uniform-marginal trios no constant/escape strategy beats E=1/3 per slot), so
            # GRPO normalization reinforces better classification, never an orthogonal cue.
            preds = _parse_trio(_completion_text(completion))
            golds = _parse_trio_gold(answer)
            n_correct = sum(1 for p, g in zip(preds, golds) if p and p == g)
            frac = n_correct / 3.0
            if require_reason and not _has_substantive_reason(_completion_text(completion), min_reason_words):
                return 0.0
            return frac
        if _is_pair_answer(answer) or info.get("task_type") == "pair":
            pa, pb = _parse_pair(_completion_text(completion))
            ga, gb = _parse_pair_gold(answer)
            correct = bool(pa and pb and pa == ga and pb == gb)
        else:
            correct = parser.parse_answer(completion) == answer
        if not correct:
            return 0.0
        if require_reason and not _has_substantive_reason(_completion_text(completion), min_reason_words):
            return 0.0
        return 1.0

    def parsed_ok(completion, **kwargs) -> float:
        """Diagnostic only (weight 0): did we get a parseable label at all?"""
        return 1.0 if parser.parse_answer(completion) else 0.0

    def reason_ok(completion, **kwargs) -> float:
        """Diagnostic only (weight 0): does the completion carry a substantive <reason_why>?
        Its batch mean tracks deliverable health live — i.e. whether the policy is keeping its
        articulated reasoning or drifting toward the empty-reason answer-only hack."""
        return 1.0 if _has_substantive_reason(_completion_text(completion), min_reason_words) else 0.0

    # Weight-0 prediction-distribution diagnostics. Their batch means (logged by the
    # rubric) expose the model's marginal PREDICTION PRIOR per step in wandb, which is
    # exactly the quantity that oscillated/collapsed in prior runs — letting us watch
    # for prior drift live instead of only post-hoc on rollout dumps.
    def _pred_indicator(label: str):
        def fn(completion, **kwargs) -> float:
            return 1.0 if parser.parse_answer(completion) == label else 0.0
        fn.__name__ = f"pred_{label.lower()}"
        return fn

    pred_claude = _pred_indicator("CLAUDE")
    pred_chatgpt = _pred_indicator("CHATGPT")
    pred_gemini = _pred_indicator("GEMINI")

    diag_funcs = [parsed_ok, reason_ok, pred_claude, pred_chatgpt]
    if label_scheme == "provider3":
        diag_funcs.append(pred_gemini)

    # Reward is correctness ONLY (weight 1.0). All other funcs are weight-0 diagnostics.
    # We deliberately do NOT add a format/valid-answer reward: under GRPO group
    # normalization a "small" additive format reward becomes full-strength in all-wrong
    # groups whose only variance is parse success, reinforcing arbitrary wrong-but-parsed
    # answers (the same failure mode as a shaped cost-matrix reward).
    rubric = vf.Rubric(
        funcs=[correct_answer, *diag_funcs],
        weights=[1.0, *([0.0] * len(diag_funcs))],
    )

    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
