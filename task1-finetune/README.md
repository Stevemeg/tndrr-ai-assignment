# Task 1 — LoRA fine-tuning: plain English → meeting/reminder JSON

Fine-tune `Qwen/Qwen2.5-0.5B-Instruct` with LoRA so that it turns a request like

> *Schedule a budget review with Priya and Rahul tomorrow at 3pm for 45 mins.*

into exactly this JSON format:

```json
{"action": "schedule_meeting", "title": "budget review", "date": "2026-11-21", "time": "15:00", "duration_minutes": 45, "attendees": ["Priya", "Rahul"]}
```

`action` is one of `schedule_meeting` / `set_reminder` / `cancel_meeting`. `date` is `YYYY-MM-DD` or null, `time` is 24-hour `HH:MM` or null, `duration_minutes` is an int or null, and `attendees` is a list of names.

## Files

| File | What it is |
|---|---|
| `generate_data.py` | Builds the synthetic dataset → `data/train.jsonl` (400), `data/test.jsonl` (60) |
| `task1_finetune.ipynb` | Kaggle notebook: base eval → LoRA training → fine-tuned eval → results |
| [`memory_calc.md`](memory_calc.md) | GPU memory estimate for fine-tuning a 7B model (full FT vs LoRA vs QLoRA) |
| `results/` | Produced by the notebook (see below) |

## How to run

**1. Generate the data locally** (standard library only):

```bash
cd task1-finetune
python generate_data.py
```

**2. Upload the data to Kaggle.** Go to kaggle.com → *Datasets* → *New Dataset*, upload `data/train.jsonl` and `data/test.jsonl`, and create it (private is fine).

**3. Open the notebook on Kaggle.** Go to *Code* → *New Notebook* → *File* → *Import Notebook* and upload `task1_finetune.ipynb`.

**4. Settings** (right-hand panel):
* *Add Input* → choose the dataset from step 2. The notebook finds `train.jsonl` anywhere under `/kaggle/input/`.
* *Accelerator* → **GPU T4 x2**. The notebook uses only one of the two GPUs.
* *Internet* → **On**. This is needed for `pip install` and for downloading the model from Hugging Face.

**5. Run all cells** (*Run All*, or *Save Version → Save & Run All*). The run downloads the ~1 GB model, evaluates the base model, trains 150 steps and evaluates again. Pip may print dependency-conflict warnings for other preinstalled Kaggle packages; they do not affect this notebook.

**6. Download the results.** Download `results.zip` from the *Output* panel, unzip it into `task1-finetune/results/`, and fill in the tables below from `results/metrics.json` and `results/comparison.md`.

### What the notebook saves in `results/`

| File | Content |
|---|---|
| `base_predictions.jsonl` | Raw output, parsed JSON and per-field correctness of the **base** model for each test example |
| `finetuned_predictions.jsonl` | Same for the **fine-tuned** model |
| `metrics.json` | Base vs fine-tuned metrics, plus training info (steps, loss, time, **peak GPU memory**) |
| `failures.jsonl` | Test examples the fine-tuned model still gets wrong, with the wrong fields listed |
| `train_loss_log.json`, `train_loss.png` | Training loss every 5 steps |
| `comparison.md` | The comparison table printed at the end of the notebook |

## Dataset design (`generate_data.py`)

* **Labels are correct by construction.** For each example the script first picks the values (action, title, date, …) and then writes them into a sentence template.
* **Fixed reference date: Friday 2026-11-20.** "Tomorrow", "in 5 days" and "on Monday" therefore each have exactly one correct date. The same date is written in the system prompt.
* **Variety.** 21 meeting titles, 20 reminder titles and 24 names. Dates are written as `today`, `tomorrow`, `the day after tomorrow`, `in N days`, `on Monday`, `on 5th Dec`, `on December 5`, `on 2026-12-05` and similar. Times are written as `3pm`, `10:30 am`, `14:30` or `noon`. Durations are written as `45 mins`, `an hour and a half`, `1.5 hours` and similar.
* **Conventions that keep labels unambiguous:**
  * A weekday name means the next such day. "On Friday" is never generated, because said on a Friday it could mean today or next week.
  * Explicit dates are at most 60 days ahead, so January dates belong to 2027.
  * A field that is not mentioned is `null` (attendees `[]`). For example, "Remind me at 5pm to …" has `date: null`.
* **Generalisation test.** The 60 test examples are 30 written with the 16 training templates (new random values) and 30 written with 6 templates that never appear in training, such as *"Pencil in a … ; 45 mins should be enough"* or *"At 3pm tomorrow, nudge me to …"*. No test sentence appears in training. Exact match is reported separately for the two groups.
* Fixed seed (`SEED = 42`), so the files are identical on every run.

## Prompt and evaluation

* **Same prompt for both models.** Both use Qwen's chat template with the same system prompt (schema, today's date, null rules) and the user's request.
* **Decoding.** Greedy (`do_sample=False`, `repetition_penalty=1.0`) with `max_new_tokens=128`.
* **Valid JSON.** The text from the first `{` to the last `}` parses as a JSON object. This tolerates a ```` ```json ```` fence, and the same rule applies to both models.
* **Exact match.** The parsed object equals the label: the same 6 keys with the same values, compared as strict string/int/list equality with no normalisation.
* **Per-field accuracy.** For each key, the share of the 60 examples where that key is present and correct. Invalid JSON counts as wrong for every field.

## LoRA settings and why

| Setting | Value | Reason |
|---|---|---|
| Rank `r` | 16 | Kept the rank small because the task is pretty narrow (structured extraction), and a bigger rank would just overfit on a small dataset (400 examples). |
| `lora_alpha` | 32 | Scale alpha/r = 2. |
| `lora_dropout` | 0.05 | Light dropout on the adapter input, a little regularisation for a small dataset. |
| Target modules | `q_proj, k_proj, v_proj, o_proj` | The attention layers, which handle how the model reads the request and picks out the parts it needs. |
| Learning rate | 2e-4 | Standard for LoRA. It's higher than for full fine-tuning because only the small adapters are trained, not the whole model. |
| Epochs | 3 | 150 steps. Enough, since the loss went down to ~0.001. More epochs would only memorise the templates further. |
| Batch size | 8 | 8 short sequences (~200–250 tokens) fit on a T4 for a 0.5B model, so no gradient accumulation is needed. If you run out of memory, use batch 4 × accumulation 2. |
| Precision | fp16 | A T4 has no bf16. The frozen base weights are fp16 (~1 GB) and the trainable LoRA weights are fp32, so the fp16 gradient scaler can update them stably. |
| Loss masking | answer only | Prompt tokens get label `-100`, which cross-entropy ignores, so the model learns to produce the JSON and the end token, not to repeat the prompt. |

Why LoRA and not full fine-tuning: I only trained lightweight adapter weights (2.16M params, 0.44%) instead of the whole model, so it fits in GPU memory, trains fast (59 s) and doesn't wreck the base model's general knowledge.

## Results

From `results/metrics.json` / `results/comparison.md` (60 test examples, greedy decoding):

| Metric | Base | Fine-tuned |
|---|---|---|
| Valid JSON rate | 100.0% | 100.0% |
| Exact match (all 60) | 0.0% | 86.7% |
| Exact match (30 seen templates) | 0.0% | 96.7% |
| Exact match (30 unseen templates) | 0.0% | 76.7% |
| Field accuracy: action | 85.0% | 95.0% |
| Field accuracy: title | 5.0% | 96.7% |
| Field accuracy: date | 38.3% | 95.0% |
| Field accuracy: time | 73.3% | 100.0% |
| Field accuracy: duration_minutes | 90.0% | 98.3% |
| Field accuracy: attendees | 48.3% | 98.3% |

The fine-tuned model still gets 8 of the 60 test examples wrong (`results/failures.jsonl`): 7 use unseen templates, 1 uses a seen template.

**Training run**

| | Value |
|---|---|
| GPU | Tesla T4 |
| Trainable parameters | 2,162,688 of 496,195,456 (0.44 %) |
| LoRA | r = 16, alpha = 32, dropout = 0.05, target modules q_proj, k_proj, v_proj, o_proj |
| Epochs / batch size / learning rate | 3 / 8 / 2e-4 |
| Optimizer steps | 150 |
| Average training loss | 0.015 |
| Last logged training loss (step 150) | 0.0007 |
| Training time | 59.0 s |
| Peak GPU memory (allocated / reserved) | 6.96 GB / 14.04 GB |

Loss curve: `results/train_loss.png` (logged every 5 steps in `results/train_loss_log.json`: 0.1521 at step 5, 0.0007 at step 150)

**What I observe:**

* Exact match went from 0.0% (base) to 86.7% (fine-tuned). Both models always produced valid JSON.
* The fine-tuned model is better on seen templates (96.7%) than on unseen ones (76.7%): 7 of the 8 remaining errors use unseen templates. As the failures below show, the fine-tune overfit a bit to the typical phrasing patterns.

## Failure analysis (3 cases from `results/failures.jsonl`)

### Failure 1: "ping me" → wrong action (test-030, 043, 058)

test-030 (unseen template), wrong field: `action`
* Input: "Ping me the day after tomorrow at 5:30pm so I remember to order a birthday cake."
* Expected: `{"action": "set_reminder", "title": "order a birthday cake", "date": "2026-11-22", "time": "17:30", "duration_minutes": null, "attendees": []}`
* Predicted: `{"action": "schedule_meeting", "title": "order a birthday cake", "date": "2026-11-22", "time": "17:30", "duration_minutes": null, "attendees": []}`

Explanation:
* The model predicted `schedule_meeting` instead of `set_reminder`, and all the other fields were right. This happened on all 3 "ping me" test sentences (test-030, 043, 058), so it is systematic, not random.
* In training, all 129 reminder sentences had the word "remind" or "reminder", and no sentence had "ping". So the model learned "remind = reminder" as a keyword shortcut instead of the meaning. With no trigger word, it fell back to `schedule_meeting`.
* Fix: more varied ways of asking for each action (e.g. paraphrase the training data with a bigger LLM).

### Failure 2: date maths (test-033, test-036)

test-033 (seen template), wrong field: `date`
* Input: "Set a reminder to send the invoice to Acme at 6:30pm in 12 days."
* Expected: `{"action": "set_reminder", "title": "send the invoice to Acme", "date": "2026-12-02", "time": "18:30", "duration_minutes": null, "attendees": []}`
* Predicted: `{"action": "set_reminder", "title": "send the invoice to Acme", "date": "2026-12-01", "time": "18:30", "duration_minutes": null, "attendees": []}`

Explanation:
* test-033: "in 12 days" from Nov 20 → the model said 2026-12-01; the correct date is 2026-12-02. Crossing into a new month broke it.
* test-036: "on Monday" → the model said 2026-11-22, which is a Sunday. The correct date is 2026-11-23.
* test-033 uses a SEEN template, so this is a reasoning weakness, not a phrasing problem.
* The model doesn't actually calculate dates; it just predicts likely-looking date tokens. Small models are bad at calendar arithmetic.
* Fix: let the model output the relative phrase (like "+12 days" or "next Monday") and calculate the real date in Python with `datetime`.

### Failure 3: subject-first sentence (test-052, test-034)

test-052 (unseen template), wrong fields: `title`, `attendees`
* Input: "The design review with Sneha in 4 days is not happening anymore, please remove it."
* Expected: `{"action": "cancel_meeting", "title": "design review", "date": "2026-11-24", "time": null, "duration_minutes": null, "attendees": ["Sneha"]}`
* Predicted: `{"action": "cancel_meeting", "title": "the design review", "date": "2026-11-24", "time": null, "duration_minutes": null, "attendees": []}`

Explanation:
* test-052: the title came out as "the design review" (extra "the"), and the attendee Sneha was dropped.
* test-034 (same pattern): the same "the design review" title error, and the date "today" was missed (it came out null).
* In all training sentences the meeting comes after the verb ("Cancel the X..."), so the model learned to find fields by their position after the verb. When the meeting is the subject of the sentence, that breaks.
* This shows the fine-tune overfit a bit to the typical phrasing patterns. It needs more varied sentence structures in training.

## Memory estimate for a 7B model

See [`memory_calc.md`](memory_calc.md).
