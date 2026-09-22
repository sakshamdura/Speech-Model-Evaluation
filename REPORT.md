# Speech Recognition Evaluation: Whisper (`base`) under Varied Conditions

**Author:** Saksham Dura | **Model:** faster-whisper `base`, CPU int8 | **Hardware:** [fill in: e.g. Windows laptop, Intel i5, 8GB RAM]

## 1. Goal
Measure how well an open-source speech recognition model (Whisper `base`) handles background noise, speaking speed, and difficult content (names, numbers, technical terms), and identify its failure patterns — including on non-speech audio.

## 2. Method
- **Dataset:** 16 clips, 3–10 seconds each, recorded by the author on a laptop microphone.
- **Ground truth:** transcripts written by hand before running the model.
- **Normalization:** lowercase, punctuation removed, whitespace collapsed.
- **Metrics:**
  - Word Error Rate (WER) = (substitutions + deletions + insertions) / reference words
  - Real-Time Factor (RTF) = processing time / audio duration
  - Hallucination count on silence/noise-only clips (no reference text)
- **Reproduce:** `python evaluate.py manifest.csv --audio-dir audio --model base`

## 3. Test cases

| ID | Condition | What it checks | Clips |
|----|-----------|----------------|-------|
| TC-01 | Quiet room, normal pace | Baseline accuracy | q1–q4 |
| TC-02 | Background noise (fan, street) | Noise robustness | n1–n3 |
| TC-03 | Fast speech | Speed robustness | f1–f2 |
| TC-04 | Slow, clear speech | Upper-bound accuracy | s1–s2 |
| TC-05 | Numbers, names, technical terms | Known weak spots | h1–h3 |
| TC-06 | Silence / noise only | Hallucination on no speech | sil1–sil2 |

## 4. Results

**Overall:** corpus WER = **18.0%** (20 errors / 111 reference words), mean RTF = **0.20** (roughly 5x faster than real time)

| Condition | Clips | Mean WER | Notes |
|-----------|-------|----------|-------|
| Baseline (quiet) | q1–q4 | 13.6% | 3 of 4 clips had at least one error |
| Noise (fan/street) | n1–n3 | 0% | See methodology note below — same sentences as baseline scored perfectly |
| Fast speech | f1–f2 | 7.1% | 1 substitution across 2 clips |
| Slow speech | s1–s2 | 0% | Perfect on both clips |
| Hard content (names/numbers/tech terms) | h1–h3 | 51.3% | By far the largest error source |
| Silence / noise-only | sil1–sil2 | n/a (WER undefined) | 1 of 2 clips hallucinated speech from noise |

**Methodology note:** grouping by condition tag alone is misleading here, because all three "hard content" clips happen to be tagged `quiet` — so a naive by-tag comparison makes noisy conditions look *more* accurate than quiet ones. The fairer test is same-sentence pairs:

| Sentence | Quiet | Noisy (fan/street) |
|---|---|---|
| "Please schedule the meeting..." | 28.6% WER | 0% WER |
| "The server restarted..." | 0% WER | 0% WER |
| "Can you send me the report..." | 10% WER | 0% WER |

On these matched pairs, background noise did not measurably hurt accuracy at this sample size.

## 5. Failure analysis

| Error type | Example (reference → model output) | Frequency |
|-----------|-------------------------------------|-----------|
| Proper nouns / names | "Saksham Dura" → "Soxium-Durak"; "Tribhuvan University" → "Tribuon University" | 1 clip, 3 word errors |
| Technical / compound terms | "PostgreSQL" → "Postgres SQL" | 1 clip, 1 substitution + 1 insertion |
| Number format mismatch | Spoken digits ("nine eight zero...") → written as numerals ("9804256256") | 1 clip, 9 of 10 errors — arguably a scoring/normalization issue rather than a true recognition failure (see note below) |
| Minor word substitution under speed/noise | "schedule" → "send"; "server" → "solver" | 3 clips, 1 error each |
| Hallucination on non-speech audio | Fan noise only → "I'm sorry, I'm sorry." | 1 of 2 silence clips |

**Normalization note on the number clip (h1):** the model transcribed the phone number correctly as digits, but the reference text spelled it out as words. This inflated the WER for that clip to 76.9%, most of which reflects a mismatch in how numbers were written down rather than a genuine transcription error. A more rigorous evaluation would normalize both reference and hypothesis to a single number format (e.g. convert both to digits) before scoring.

## 6. Limitations
- Only 16 clips from a single speaker — not enough to draw statistically reliable conclusions, only directional ones.
- The noise and "hard content" conditions were not fully isolated from each other (see methodology note above), so the noise-robustness result should be treated cautiously.
- Reference transcripts were written by the same person who recorded the clips, with no independent verification.
- Only one model size (`base`) was tested; larger models may perform differently, especially on proper nouns.

## 7. Recommendations
- **Names and technical terms are the model's weakest area** (51.3% WER vs. 13.6% baseline). If this model were used in a product context, add a custom vocabulary list or post-processing correction step for known proper nouns and domain terms.
- **Fix the number-formatting mismatch** before scoring any clip with spoken numbers — normalize both reference and hypothesis to digits (or both to words) rather than comparing across formats.
- **Investigate the hallucination case further**: run more noise-only clips (different noise types and volumes) to see how often and under what conditions the model invents speech from silence. This matters for any always-listening application.
- **Re-run with a larger model** (`small` or `medium`) on the same clips to see whether the proper-noun and hallucination issues improve, and whether that's worth the ~2–3x slower processing time.

## 8. Next steps
- Expand to 40–50 clips across 2–3 speakers to get statistically meaningful per-condition comparisons.
- Redesign the noise test so noise and content difficulty are varied independently (e.g. also record hard-content clips under noise).
- Add streaming/incremental latency measurement, not just batch RTF.
- Test model size vs. accuracy/latency tradeoff directly (`base` vs `small` vs `medium`) on the same clip set.
