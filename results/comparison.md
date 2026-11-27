# Week 1 and Week 2 acceptance comparison

Run `./.venv/bin/python -m pytest tests/ -v` from this checkout. A working environment on PATH also supports `pytest tests/ -v`. This checkout's generated activation script and pytest launcher point to the project's old path. Tests use temporary databases and loopback servers. The existing 15-guide index is copied read-only.

The same 10 clean prompts and five missing-fact prompts run on both models. Answers are checked for expected facts and source pages. Feature 09 measures the first nonblank content token, generation rate, and total wall time directly from warmed Ollama streams. RAM means Ollama's loaded model allocation, including context. Other times are whole test durations.

| Feature ID | Test | 1.7B | 4B | PASS/FAIL | Speed (1.7B / 4B) | RAM (1.7B / 4B) |
|---|---|---:|---:|---|---|---|
| 01 | FastAPI docs, health, and malformed body | 1/1 | 1/1 | PASS | 0.01s / 0.02s | — / — |
| 02a | Ollama answers while running | 1/1 | 1/1 | PASS | 3.23s / 6.23s | — / — |
| 02b | Clean error after the test Ollama stops | 1/1 | 1/1 | PASS | 1.03s / 1.51s | — / — |
| 03 / 06 | Same 10 IT questions with evidence and citations | 10/10 | 10/10 | PASS | 1.60s / 5.30s | — / — |
| 04 | Ordinary SQLite chat transcript survives restart | 1/1 | 1/1 | PASS | 1.00s / 1.53s | — / — |
| 05 | Answers use an uploaded document | 1/1 | 1/1 | PASS | 0.85s / 2.17s | — / — |
| 07 | Switch 1.7B → 4B → 1.7B and retain same-tab history | 1/1 | 1/1 | PASS | 11.81s / 11.81s | — / — |
| 07 UI | Visual model picker and history in a browser | MANUAL | MANUAL | MANUAL | — / — | — / — |
| 08a | Verified answer text streams incrementally | 1/1 | 1/1 | PASS | 0.93s / 2.23s | — / — |
| 08b | Cancel a live stream and use the API again | 1/1 | 1/1 | PASS | 0.82s / 1.91s | — / — |
| 09 | Warm Ollama TTFT, tokens/sec, latency and loaded memory | 10/10 | 10/10 | PASS | TTFT 0.09s, 45.1 tok/s, latency 2.16s / TTFT 0.23s, 17.8 tok/s, latency 5.68s | 2257 MiB / 3630 MiB |
| 10a | Archive is hidden and its checkpoint restores in isolation | 1/1 | 1/1 | PASS | 0.28s / 0.28s | — / — |
| 10b | Archived model configuration is saved and restorable | 1/1 | 1/1 | PASS | 0.01s / 0.01s | — / — |
| 11 | Recall a conversation-only fact after a few turns | 1/1 | 1/1 | PASS | 2.33s / 7.66s | — / — |
| 12 | Approved fact persists through restart and a new session | 1/1 | 1/1 | PASS | 1.67s / 3.80s | — / — |
| 13 | 20-turn context retains key facts and invents none | 1/1 | 1/1 | PASS | 0.50s / 1.07s | — / — |
| 14 | Semantic retrieval finds a reworded query's passage | 1/1 | 1/1 | PASS | 3.47s / 6.12s | — / — |
| 15a | Normal and multi-page PDFs | 1/1 | 1/1 | PASS | 0.46s / 1.39s | — / — |
| 15b | Empty and corrupt PDFs return clean errors | 1/1 | 1/1 | PASS | 2.74s / 7.00s | — / — |
| 15c | Image-only PDF uses OCR | 1/1 | 1/1 | PASS | 32.43s / 49.13s | — / — |
| 16a | Approved website excerpt and URL, unapproved URL rejected | 1/1 | 1/1 | PASS | 1.18s / 2.63s | — / — |
| 16b | Approved URL-only ingestion through a controlled fetch fixture | 1/1 | 1/1 | PASS | 0.07s / 0.04s | — / — |
| 17 | PDF page and Markdown section agree with the evidence | 1/1 | 1/1 | PASS | 1.81s / 4.39s | — / — |
| 18 | Update changes answer, deletion removes the evidence and citation | 1/1 | 1/1 | PASS | 1.53s / 4.32s | — / — |
| 19 | Add guide asks and honors No and Yes | 1/1 | 1/1 | PASS | 0.62s / 1.03s | — / — |
| Hallucination | Five out-of-document questions should abstain | 5/5 | 5/5 | PASS | 1.22s / 4.61s | — / — |

## Per-model summary

Pass rates count individual automated acceptance cases, including shared checks, and exclude MANUAL.

- **qwen3:1.7b** · 47/47 (100.0%) · TTFT 0.09s, 45.1 tok/s, latency 2.16s · average RAM 2257 MiB
- **qwen3:4b** · 47/47 (100.0%) · TTFT 0.23s, 17.8 tok/s, latency 5.68s · average RAM 3630 MiB

## Failures and manual checks

- **07 UI · qwen3:1.7b  · MANUAL** — Inspect the rendered controls and history in a real browser.
- **07 UI · qwen3:4b  · MANUAL** — Inspect the rendered controls and history in a real browser.

## Recommendation

- Use **qwen3:1.7b** for routine guide lookups. Both models passed 10 of the same 10 cases.
- Use **qwen3:4b** for more involved document questions when answer quality is the priority. It led the broader 40-question comparison on the same collection. [Broader benchmark](../README.md#model-comparison-on-the-m1-macbook-air)
- Use **qwen3:1.7b** when the Mac has limited available memory. See the measured loaded RAM in feature 09.
