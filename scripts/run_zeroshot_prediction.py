"""Zero-shot LLM baseline for next-action prediction.

The trained models predict what the agent will do from the developer's prompt
alone. This asks an LLM the same question, so the paper can say whether a
trained classifier beats simply asking a model.

The LLM sees ONLY what the models see: the developer prompt, optionally the
previous turn's prompt, and the prompt-derived metadata. It never sees the
agent's response, which is what the label is derived from.

Evaluated on the same repository-held-out test split as the main run, so the
numbers are comparable; --sample caps the number of API calls.

Usage:
  python scripts/run_zeroshot_prediction.py outputs/llm_labels_v4/llm_labels_full.csv \
      --output-dir outputs/ml_action_v4_llm/zeroshot --label-col llm_action_type
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
from sklearn.metrics import classification_report, f1_score

import github_repo_miner.ml_research.llm_labeling as ll
from github_repo_miner.ml_research.action_prediction import ACTION_TYPES, load_action_dataset
from github_repo_miner.ml_research.splits import repo_train_test_split

load_dotenv(ROOT / ".env")
DEFAULT_MODEL = "gemini-3.1-flash-lite"
PROMPT_CHARS, PREV_CHARS = 4000, 800
GROQ_MIN_INTERVAL = 8.0  # free tier is 8k tokens/minute per model; pace the calls
# Reasoning budget for Groq models that support it. Set GROQ_REASONING_EFFORT=medium
# to give an open model the same room to think that the Gemini models take by default.
GROQ_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")

INSTRUCTIONS = f"""You are predicting what an AI coding agent will do NEXT, before it responds.

You see a developer's message to the agent, and sometimes the developer's previous message in the
same session. You do NOT see the agent's reply — that is what you are predicting.

Predict the agent's next action as exactly one of: {", ".join(ACTION_TYPES)}.

- implement: it will produce or change project code, config, tests, docs or repository state as new work.
- debug: it will work on a problem the developer reported (an error, failing output, wrong behavior).
- explain: it will deliver understanding — concepts, how code works, options, instructions, or a plan —
  without changing the project.
- review: it will assess existing work and report findings or a status, with no problem reported.
- tool_only: it will only search, list or read files and stop before producing a result.
- other: nothing meaningful for coding (greetings, thanks, refusals, system messages).

Base the prediction on what the developer asks for and how agents typically respond to such a request.
Respond with ONLY this JSON object:
{{"action_type": "<one of {ACTION_TYPES}>", "confidence": <0.0-1.0>, "rationale": "<one sentence>"}}"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_csv", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--label-col", default=None)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--sample", type=int, default=0, help="Cap on test turns to label (0 = all).")
    p.add_argument("--with-prev-turn", action="store_true", help="Also show the previous developer message.")
    p.add_argument("--provider", choices=["gemini", "groq"], default="gemini")
    p.add_argument("--model", default=None, help=f"Default: {DEFAULT_MODEL} for gemini.")
    p.add_argument("--max-workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_prompt(row: pd.Series, with_prev: bool) -> str:
    parts = [INSTRUCTIONS, "", "Now predict this turn:", ""]
    if with_prev and pd.notna(row.get("prev_prompt")):
        parts.append(f"Previous developer message: {str(row['prev_prompt'])[:PREV_CHARS]!r}")
    parts += [
        f"Developer message: {str(row['prompt_text'])[:PROMPT_CHARS]!r}",
        f"Metadata: words={int(row.get('prompt_word_count', 0) or 0)}, "
        f"turns_before={int(row.get('turns_before', 0) or 0)}, "
        f"previous_turn_had_code={bool(row.get('prev_turn_had_code', False))}, "
        f"previous_turn_had_tools={bool(row.get('prev_turn_had_tools', False))}",
    ]
    return "\n".join(parts)


def predict_one(text: str, provider: str = "gemini", model: str | None = None) -> str | None:
    """One prediction, retrying transient failures. Returns None if it never parses."""
    model = model or (DEFAULT_MODEL if provider == "gemini" else "openai/gpt-oss-120b")
    for _ in range(5):
        try:
            if provider == "gemini":
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
                raw = client.models.generate_content(
                    model=model, contents=text,
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
                ).text
            else:
                from groq import Groq
                extra = {"reasoning_effort": GROQ_EFFORT} if "gpt-oss" in model else {"reasoning_format": "hidden"}
                raw = Groq(api_key=os.environ["GROQ_API_KEY"]).chat.completions.create(
                    model=model, messages=[{"role": "user", "content": text}], temperature=0.0,
                    response_format={"type": "json_object"},
                    max_completion_tokens=800 if GROQ_EFFORT == "low" else 2500, **extra,
                ).choices[0].message.content
            res = ll._parse_response(raw or "")
            if not res.error:
                return res.action_type
        except Exception as exc:
            msg = str(exc).lower()
            if "per day" in msg or "tpd" in msg or "rpd" in msg:
                raise RuntimeError("daily quota reached") from exc
            time.sleep(GROQ_MIN_INTERVAL if provider == "groq" else 8)
            continue
        time.sleep(4)
    return None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df, label_col = load_action_dataset(args.input_csv, label_col=args.label_col)
    df = ll.attach_previous_turn(df)
    _, test_idx = repo_train_test_split(df, test_size=args.test_size, random_state=args.seed)
    test = df.iloc[test_idx]
    if args.sample and args.sample < len(test):
        test = test.sample(args.sample, random_state=args.seed)
    print(f"test turns: {len(test)} from {test['full_name'].nunique()} repositories; target {label_col}")

    cache_path = args.output_dir / "zeroshot_predictions.csv"
    done = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["row", "pred"])
    todo = test[~test.index.isin(done["row"])]
    print(f"to predict: {len(todo)} (cached: {len(done)})")
    if len(todo):
        rows = [r for _, r in todo.iterrows()]
        preds: list[str | None] = []
        try:
            if args.provider == "groq":  # serial: the free tier allows only a few calls a minute
                for i, r in enumerate(rows, 1):
                    preds.append(predict_one(build_prompt(r, args.with_prev_turn), args.provider, args.model))
                    if i % 20 == 0:
                        pd.concat([done, pd.DataFrame({"row": todo.index[:len(preds)], "pred": preds})],
                                  ignore_index=True).to_csv(cache_path, index=False)
                        print(f"  {i}/{len(rows)}", flush=True)
                    time.sleep(GROQ_MIN_INTERVAL)
            else:
                with ThreadPoolExecutor(args.max_workers) as ex:
                    preds = list(ex.map(
                        lambda r: predict_one(build_prompt(r, args.with_prev_turn), args.provider, args.model), rows))
        except RuntimeError as exc:  # daily quota: keep what we have
            print(f"stopped after {len(preds)} predictions: {exc}", flush=True)
        done = pd.concat([done, pd.DataFrame({"row": todo.index[:len(preds)], "pred": preds})], ignore_index=True)
        done.to_csv(cache_path, index=False)

    merged = test.join(done.set_index("row")["pred"], how="left")
    ok = merged[merged["pred"].notna()]
    y, p = ok[label_col].to_numpy(), ok["pred"].to_numpy()
    macro = f1_score(y, p, average="macro", zero_division=0)
    rep = classification_report(y, p, labels=ACTION_TYPES, output_dict=True, zero_division=0)
    model_name = args.model or (DEFAULT_MODEL if args.provider == "gemini" else "openai/gpt-oss-120b")
    lines = ["# Zero-shot LLM baseline for next-action prediction\n",
             f"Model `{model_name}` ({args.provider}), temperature 0, prompt-side inputs only"
             f"{' plus the previous developer message' if args.with_prev_turn else ''}.\n",
             f"- target: `{label_col}`",
             f"- test turns scored: {len(ok)} of {len(merged)} (failed calls: {int(merged['pred'].isna().sum())})",
             f"- accuracy: {float((y == p).mean()):.1%}",
             f"- macro-F1: {macro:.4f}", "",
             "| label | precision | recall | support |", "|---|---|---|---|"]
    lines += [f"| {k} | {rep[k]['precision']:.2f} | {rep[k]['recall']:.2f} | {int(rep[k]['support'])} |"
              for k in ACTION_TYPES]
    (args.output_dir / "ZEROSHOT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
