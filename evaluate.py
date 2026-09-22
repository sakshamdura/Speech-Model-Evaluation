"""
Evaluate a speech-to-text model (faster-whisper) on your own audio clips.

Measures, per clip:
  - Word Error Rate (WER) plus substitutions / deletions / insertions
  - Latency: processing time and real-time factor (RTF = processing time / audio length)

Usage:
    python evaluate.py manifest.csv --audio-dir audio --model base

manifest.csv needs at least the columns `file` and `reference`.
Optional: `language` (default "en"; use "ne" for Nepali).
Every other column (speaker, noise, speed, ...) is treated as a condition tag
and gets its own summary table.

Leave `reference` empty for silence / noise-only clips. The script then counts
how many words the model "hallucinated" instead of computing WER.
"""
import argparse
import string
import time
from pathlib import Path

import jiwer
import pandas as pd
from faster_whisper import WhisperModel

# Delete ASCII punctuation plus the Devanagari danda so Nepali references work too.
PUNCT_TABLE = str.maketrans("", "", string.punctuation + "।")
RESERVED_COLUMNS = {"file", "reference", "language"}


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = str(text).lower().translate(PUNCT_TABLE)
    return " ".join(text.split())


def score(ref: str, hyp: str) -> dict:
    """Word-level error counts between normalized reference and hypothesis."""
    ref_words, hyp_words = ref.split(), hyp.split()
    if not ref_words:  # silence clip: any output is a hallucination
        return {"ref_words": 0, "wer": None, "subs": 0, "dels": 0, "ins": len(hyp_words)}
    if not hyp_words:  # model returned nothing
        return {"ref_words": len(ref_words), "wer": 1.0, "subs": 0, "dels": len(ref_words), "ins": 0}
    out = jiwer.process_words(ref, hyp)
    return {
        "ref_words": len(ref_words),
        "wer": out.wer,
        "subs": out.substitutions,
        "dels": out.deletions,
        "ins": out.insertions,
    }


def transcribe(model: WhisperModel, path: Path, language: str):
    """Return (text, audio_seconds, processing_seconds)."""
    start = time.perf_counter()
    segments, info = model.transcribe(str(path), language=language, beam_size=5)
    # `segments` is a lazy generator: the real work happens while we consume it,
    # so the timer must stop AFTER this line.
    text = " ".join(seg.text.strip() for seg in segments)
    elapsed = time.perf_counter() - start
    return text, info.duration, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", help="CSV with columns: file, reference, [language], [tags...]")
    parser.add_argument("--audio-dir", default="audio")
    parser.add_argument("--model", default="base", help="tiny | base | small | medium ...")
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()

    # keep_default_na=False so an empty reference stays "" instead of becoming NaN
    df = pd.read_csv(args.manifest, keep_default_na=False)
    missing = {"file", "reference"} - set(df.columns)
    if missing:
        raise SystemExit(f"manifest is missing required columns: {sorted(missing)}")
    if "language" not in df.columns:
        df["language"] = "en"
    df["language"] = df["language"].replace("", "en")

    print(f"Loading model '{args.model}' (CPU, int8)...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    # Warm-up run so model loading / first-call overhead doesn't pollute latency numbers.
    first = df.iloc[0]
    transcribe(model, Path(args.audio_dir) / first["file"], first["language"])

    rows = []
    for _, row in df.iterrows():
        path = Path(args.audio_dir) / row["file"]
        hyp_raw, audio_sec, proc_sec = transcribe(model, path, row["language"])
        ref, hyp = normalize(row["reference"]), normalize(hyp_raw)
        result = {
            **row.to_dict(),
            "hypothesis": hyp_raw,
            **score(ref, hyp),
            "audio_sec": round(audio_sec, 2),
            "proc_sec": round(proc_sec, 2),
            "rtf": round(proc_sec / audio_sec, 3) if audio_sec else None,
        }
        rows.append(result)
        wer_txt = "n/a (silence)" if result["wer"] is None else f"{result['wer']:.2f}"
        print(f"{row['file']:<28} WER={wer_txt:<14} RTF={result['rtf']}")

    res = pd.DataFrame(rows)
    res["wer"] = pd.to_numeric(res["wer"])
    res.to_csv(args.out, index=False)

    # ---- Summary -------------------------------------------------------
    scored = res[res["ref_words"] > 0]
    total_errors = (scored["subs"] + scored["dels"] + scored["ins"]).sum()
    corpus_wer = total_errors / scored["ref_words"].sum()
    print("\n=== Overall ===")
    print(f"Model: {args.model} | clips: {len(res)} | corpus WER: {corpus_wer:.3f}")
    print(f"Mean RTF: {res['rtf'].mean():.3f}  (below 1.0 = faster than real time)")

    for tag in [c for c in df.columns if c not in RESERVED_COLUMNS]:
        print(f"\n=== By {tag} ===")
        table = res.groupby(tag).agg(
            clips=("file", "count"),
            mean_wer=("wer", "mean"),
            mean_rtf=("rtf", "mean"),
        )
        print(table.round(3))

    print("\n=== 5 worst clips by WER ===")
    print(scored.sort_values("wer", ascending=False)[["file", "wer", "reference", "hypothesis"]].head(5).to_string(index=False))

    silent = res[res["ref_words"] == 0]
    if not silent.empty:
        print("\n=== Silence / noise-only clips ===")
        print(silent[["file", "ins", "hypothesis"]].to_string(index=False))

    print(f"\nFull per-clip results saved to {args.out}")


if __name__ == "__main__":
    main()
