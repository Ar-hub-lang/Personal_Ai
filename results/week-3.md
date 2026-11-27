# Week 3 verification

Tested on November 27, 2026 against the running local IT Support app.

| Feature | Diagnostic | Live result |
| --- | --- | --- |
| 20 | Hardware and OS | PASS |
| 21 | CPU | PASS |
| 22 | RAM | PASS |
| 23 | Disk space | PASS |
| 24 | Network routes | PASS |
| 25 | DNS lookup | PASS |
| 26 | TCP port usage | PASS |
| 27 | Processes | PASS |
| 28 | System log counts | PASS |
| 29 | Git | PASS |
| 30 | Python environment | PASS |
| 31 | Node.js and npm | PASS |
| 32 | SQLite connectivity | PASS |
| 33 | Docker engine | PASS |

Docker Desktop was started for the test. Its local engine responded with version 29.8.0 and API 1.56.
The TCP test identified the owner of a temporary listener and reported the port free after it closed.
Invalid check requests and requests from an unrelated browser origin were rejected.

## Regression tests through Week 3

Across the full run and focused reruns, all 173 automated cases passed.

The first run passed 169 cases. The two streaming cases expected multiple chunks from a one-word answer. The revised test checks short and multiword answers and verifies that the chunks reconstruct the cited final reply. Both models passed.

The legacy TinyGPT archive was absent from the current checkout’s parent directory. Both archive checks passed on the retained test copy.

| Model | Cited IT answers | Missing-fact questions that abstained |
| --- | --- | --- |
| Qwen3 1.7B | 10/10 | 5/5 |
| Qwen3 4B | 10/10 | 5/5 |

The regression suite used a separate 15-guide index and temporary chat databases.
Safari opened the Diagnostics panel. The 14 system checks ran through the live API.
The earlier visual model-switch and history check remains a manual item in the acceptance report.
