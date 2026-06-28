"""Pre-flight headroom diagnostic for OPCD/RLSD (experiments E2/E3).

Before spending 8xH100 on on-policy distillation we must confirm the regime is
non-degenerate: the cheatsheet TEACHER (same policy, cheatsheet in-context) must
be meaningfully BETTER than the PLAIN student on the eval, otherwise the
distillation signal is empty (teacher ~= student) and the run is uninformative.

This script measures, with NO training, on a sample of a split:

  * student_acc / teacher_acc          - argmax-over-labels accuracy
  * student_gold_p / teacher_gold_p    - mean softmax prob on the gold label
  * kl_teacher_student                 - mean KL(teacher || student) over the 3 labels
  * frac_teacher_beats_student         - frac of rows where teacher_gold_p > student_gold_p

Method (faithful to the orchestrator's teacher scoring): for each candidate label
L we render ``<answer>L</answer>`` as a forced completion, score it with vLLM
``prompt_logprobs`` summed over the completion tokens, and softmax the three
sequence-logprobs into a label distribution. The PLAIN prompt = the rollout
prompt (student); the CHEAT prompt = PLAIN + CHEATSHEET_TRAIN (teacher). This is
exactly the conditioning gap the on-policy distillation loss exploits.

This is a label-discriminability probe (minimal forced answer, no reasoning),
not the full rollout distribution. It is intentionally cheap so it can gate the
real run. Run on the SAME data/split the experiment will use:

    uv run python scripts/diag_teacher_student.py --split val --n 200
"""

import argparse
import math
import sys
from pathlib import Path

from datasets import load_from_disk

ENV_DIR = Path(__file__).resolve().parent.parent / "deps/research-environments/environments/blog_author_id"
sys.path.insert(0, str(ENV_DIR))
from blog_author_id import CHEATSHEET_TRAIN, SYSTEM_PROMPT_3WAY  # noqa: E402

MODEL = "Qwen/Qwen3.5-9B"
DATA = Path(__file__).resolve().parent.parent / "data/blog_author_id_3way_v2"
LABELS = ["CLAUDE", "CHATGPT", "GEMINI"]


def _softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    z = sum(es)
    return [e / z for e in es]


def _kl(p, q):
    # KL(p || q) with a small floor to avoid log(0).
    eps = 1e-12
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q))


def render_prompt(tok, system, blog):
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": blog}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "val_ood", "train"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gpu_mem", type=float, default=0.5)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams  # imported lazily (GPU dependency)

    ds = load_from_disk(str(DATA / args.split))
    n = min(args.n, len(ds))
    rows = [ds[i] for i in range(n)]

    llm = LLM(model=args.model, trust_remote_code=True,
              gpu_memory_utilization=args.gpu_mem, enforce_eager=True,
              max_model_len=16384)
    tok = llm.get_tokenizer()

    completions = {L: f"<answer>{L}</answer>" for L in LABELS}
    # Score one forced completion per (row, prompt-variant, label) via prompt_logprobs.
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)

    def seq_logprob(prompt_text, completion_text):
        full = prompt_text + completion_text
        prompt_len = len(tok.encode(prompt_text, add_special_tokens=False))
        out = llm.generate([full], sp, use_tqdm=False)[0]
        plps = out.prompt_logprobs  # list[dict|None], one per prompt token
        total = 0.0
        for pos in range(prompt_len, len(plps)):
            d = plps[pos]
            if not d:
                continue
            tid = next(iter(d))  # the actual token at this position
            total += d[tid].logprob
        return total

    agg = {
        "student_correct": 0, "teacher_correct": 0,
        "student_gold_p": 0.0, "teacher_gold_p": 0.0,
        "kl": 0.0, "teacher_beats": 0,
    }
    for r in rows:
        blog, gold = r["question"], r["answer"]
        plain = render_prompt(tok, SYSTEM_PROMPT_3WAY, blog)
        cheat = render_prompt(tok, SYSTEM_PROMPT_3WAY + CHEATSHEET_TRAIN, blog)
        s_lp = [seq_logprob(plain, completions[L]) for L in LABELS]
        t_lp = [seq_logprob(cheat, completions[L]) for L in LABELS]
        p_s, p_t = _softmax(s_lp), _softmax(t_lp)
        gi = LABELS.index(gold)
        agg["student_correct"] += int(max(range(3), key=lambda k: p_s[k]) == gi)
        agg["teacher_correct"] += int(max(range(3), key=lambda k: p_t[k]) == gi)
        agg["student_gold_p"] += p_s[gi]
        agg["teacher_gold_p"] += p_t[gi]
        agg["kl"] += _kl(p_t, p_s)
        agg["teacher_beats"] += int(p_t[gi] > p_s[gi])

    print(f"\n=== teacher-vs-student headroom  split={args.split}  n={n}  model={args.model} ===")
    print(f"student_acc            {agg['student_correct'] / n:.3f}")
    print(f"teacher_acc            {agg['teacher_correct'] / n:.3f}")
    print(f"student_gold_p (mean)  {agg['student_gold_p'] / n:.3f}")
    print(f"teacher_gold_p (mean)  {agg['teacher_gold_p'] / n:.3f}")
    print(f"KL(teacher||student)   {agg['kl'] / n:.3f}")
    print(f"frac teacher>student   {agg['teacher_beats'] / n:.3f}")
    print("\nGO criterion: teacher_acc and teacher_gold_p clearly > student, KL > 0,")
    print("frac teacher>student well above 0.5 => non-degenerate distillation signal.")


if __name__ == "__main__":
    main()
