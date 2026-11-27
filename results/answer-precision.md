# IT Support answer precision

Measured November 26, 2026 against the frozen 15-guide pilot collection. The
same 30 tuning questions with spelling variations and 10 held-out questions
were used for both models. A pass requires the expected facts and source pages.

| Model | Baseline | Full live run after focusing evidence | Final assembly | Retrieval hits | Unsupported answers on missing facts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3 4B | 36/40 | 39/40 | 40/40 | 31/31 | 0/9 |
| Qwen3 1.7B | 30/40 | 30/40 | 30/40 | 31/31 | 0/9 |

The final assembly processes the same model responses with the added rule for
questions about starting a procedure. That rule fixed the remaining 4B tuning
case by removing a step that takes place after registration. A separate live
request passed that case. The final assembly was replayed on all 40 questions
for both models. Qwen3 4B passed all 10 held-out questions in the live run.

Across supported questions both versions answered correctly, the median reply
length fell from 33 to 29 words for 4B and from 31 to 26 words for 1.7B. The
assembler uses cited source text, preserves stated conditions and full requested
procedures, and leaves out an extra citation when none of its text supports the
focused reply.

## Unseen PDF smoke check

An isolated, freshly uploaded three-page printer guide tested a short fact,
an installation procedure, a question spanning two pages, and a missing fact.
Qwen3 4B passed all four checks. Qwen3 1.7B passed three. Its installation reply
skipped the Wi-Fi prerequisite and cited an unrelated page. The local Qwen3 4B
default remains the better choice for document-based troubleshooting.

## Week 3 verification

The installed FastAPI app returned results for 13 of the 14 diagnostics. The
Docker check detected the CLI and reported an unavailable local engine. Its
success response was exercised with a test fixture. The Diagnostics panel and
hardware result were verified in Safari.
