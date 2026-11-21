| Metric | qwen2.5:1.5b-instruct-fp16 | qwen2.5:1.5b-instruct-q4_K_M |
|---|---|---|
| Generation speed (tokens/s) | 12.3 | 28.1 |
| Prompt processing speed (tokens/s) | 139.9 | 224.5 |
| Total time, 10 prompts (s) | 77.6 | 35.9 |
| Output tokens, 10 prompts | 886 | 912 |
| Memory: size (GB) | 3.28 | 1.17 |
| Memory: size_vram (GB) | 0.00 | 0.00 |

| # | Category | tok/s fp16 | tok/s q4_K_M | output tokens fp16 | output tokens q4_K_M |
|---|---|---|---|---|---|
| 1 | factual | 11.7 | 31.9 | 51 | 103 |
| 2 | reasoning | 11.8 | 28.7 | 256 | 256 |
| 3 | coding | 12.4 | 28.2 | 256 | 176 |
| 4 | summarisation | 12.5 | 24.2 | 33 | 33 |
| 5 | json_output | 13.4 | 17.5 | 44 | 44 |
| 6 | creative | 12.8 | 25.2 | 41 | 138 |
| 7 | maths | 11.7 | 33.6 | 105 | 98 |
| 8 | translation | 13.2 | 37.6 | 24 | 22 |
| 9 | classification | 14.3 | 34.6 | 12 | 12 |
| 10 | instruction_following | 13.6 | 33.7 | 64 | 30 |
