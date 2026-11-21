"""Generate a synthetic dataset: plain-English meeting/reminder requests -> JSON.

Each example is built in two steps:
  1. pick the ground-truth values (action, title, date, time, duration, attendees)
  2. write them into a sentence using a phrasing template
so every label is correct by construction.

Some templates are used ONLY in the test set, so the test measures whether the
model generalises to new phrasings instead of memorising the training ones.

Run:  python generate_data.py
Writes data/train.jsonl and data/test.jsonl
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
REFERENCE_DATE = date(2026, 11, 20)  # "today" for every example (a Friday)
N_TRAIN = 400
N_TEST_SEEN = 30    # test examples written with training templates (new random values)
N_TEST_UNSEEN = 30  # test examples written with templates never used in training
OUT_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Slot values
# ---------------------------------------------------------------------------
NAMES = [
    "Priya", "Rahul", "Ankit", "Sneha", "Arjun", "Meera", "Vikram", "Kavya",
    "Rohan", "Ananya", "John", "Sarah", "David", "Emma", "Michael", "Olivia",
    "Wei", "Fatima", "Carlos", "Aisha", "Kenji", "Lucas", "Zara", "Omar",
]

# Meeting titles start with a consonant sound so "a {title}" reads correctly.
MEETING_TITLES = [
    "project sync", "budget review", "sprint planning session", "design review",
    "client call", "team standup", "quarterly review", "hiring discussion",
    "roadmap planning meeting", "product demo", "marketing kickoff", "code review",
    "vendor negotiation", "performance review", "data pipeline sync",
    "customer feedback session", "board prep meeting", "website redesign review",
    "sales pipeline review", "weekly catch-up", "one-on-one",
]

# Reminder titles are verb phrases so "remind me to {title}" reads correctly.
REMINDER_TITLES = [
    "call the dentist", "submit the expense report", "pay the electricity bill",
    "renew my passport", "send the invoice to Acme", "book flight tickets",
    "water the plants", "review the pull request", "buy groceries",
    "email the slides to the team", "back up my laptop", "pick up the dry cleaning",
    "update the project tracker", "follow up with the recruiter",
    "order a birthday cake", "check the server logs", "prepare the demo script",
    "return the library books", "file the tax documents", "confirm the hotel booking",
]

# (how it is written, value in minutes)
DURATIONS = [
    ("15 minutes", 15), ("15 mins", 15), ("half an hour", 30), ("30 minutes", 30),
    ("30 min", 30), ("45 mins", 45), ("45 minutes", 45), ("an hour", 60),
    ("1 hour", 60), ("90 minutes", 90), ("1.5 hours", 90),
    ("an hour and a half", 90), ("2 hours", 120),
]

MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ---------------------------------------------------------------------------
# Phrasing templates
# Placeholders: {title} {date} {time} {duration} {attendees}
# If a placeholder is missing from a template, that field is not mentioned,
# so its label is null (or [] for attendees).
# ---------------------------------------------------------------------------
TRAIN_TEMPLATES = {
    "schedule_meeting": [
        "Schedule a {title} with {attendees} {date} at {time} for {duration}.",
        "Set up a {title} {date} at {time}.",
        "Can you book a {title} with {attendees} {date} at {time}?",
        "Please arrange a {title} for {duration} {date} at {time}.",
        "I need a {title} with {attendees} {date}.",
        "Add a {title} to my calendar {date} at {time} with {attendees}.",
        "Book {duration} for a {title} with {attendees} {date} at {time}.",
    ],
    "set_reminder": [
        "Remind me to {title} {date} at {time}.",
        "Set a reminder to {title} at {time} {date}.",
        "Remind me to {title} {date}.",
        "Please remind me at {time} to {title}.",
        "Create a reminder {date} at {time}: {title}.",
    ],
    "cancel_meeting": [
        "Cancel the {title} {date}.",
        "Cancel my {title} with {attendees} {date} at {time}.",
        "Please call off the {title} with {attendees}.",
        "Delete the {title} at {time} {date} from my calendar.",
    ],
}

TEST_ONLY_TEMPLATES = {
    "schedule_meeting": [
        "Could you get me together with {attendees} for a {title} {date} at {time}? Plan for {duration}.",
        "Pencil in a {title} {date} at {time} with {attendees}; {duration} should be enough.",
    ],
    "set_reminder": [
        "Ping me {date} at {time} so I remember to {title}.",
        "At {time} {date}, nudge me to {title}.",
    ],
    "cancel_meeting": [
        "The {title} with {attendees} {date} is not happening anymore, please remove it.",
        "Something came up, please scrap the {title} {date} at {time}.",
    ],
}


# ---------------------------------------------------------------------------
# Random value generators. Each returns (text as written, label value).
# ---------------------------------------------------------------------------
def ordinal(n):
    """1 -> '1st', 2 -> '2nd', 11 -> '11th', 23 -> '23rd'."""
    if 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def random_date(rng):
    """A date phrase relative to REFERENCE_DATE and its 'YYYY-MM-DD' value."""
    kind = rng.choice(["today", "tomorrow", "day_after", "in_days", "weekday", "absolute"])

    if kind == "today":
        d, phrase = REFERENCE_DATE, "today"
    elif kind == "tomorrow":
        d, phrase = REFERENCE_DATE + timedelta(days=1), "tomorrow"
    elif kind == "day_after":
        d, phrase = REFERENCE_DATE + timedelta(days=2), "the day after tomorrow"
    elif kind == "in_days":
        n = rng.randint(3, 14)
        d, phrase = REFERENCE_DATE + timedelta(days=n), f"in {n} days"
    elif kind == "weekday":
        # "on Monday" means the next Monday after today. Only 1-6 days ahead,
        # so today's weekday (Friday) is never used: "on Friday" said on a
        # Friday is ambiguous (today or next week?).
        d = REFERENCE_DATE + timedelta(days=rng.randint(1, 6))
        phrase = f"on {WEEKDAYS[d.weekday()]}"
    else:
        # An explicit date in the next 60 days (dates in January are in 2027).
        d = REFERENCE_DATE + timedelta(days=rng.randint(1, 60))
        month = MONTHS[d.month - 1]
        phrase = rng.choice([
            f"on {ordinal(d.day)} {month[:3]}",  # on 5th Dec
            f"on {d.day} {month}",               # on 5 December
            f"on {month} {d.day}",               # on December 5
            f"on {month[:3]} {ordinal(d.day)}",  # on Dec 5th
            f"on {d.isoformat()}",               # on 2026-12-05
        ])
    return phrase, d.isoformat()


def random_time(rng):
    """A time phrase and its 24-hour 'HH:MM' value."""
    hour = rng.randint(8, 18)
    minute = rng.choice([0, 0, 15, 30, 45])
    value = f"{hour:02d}:{minute:02d}"

    if hour == 12 and minute == 0 and rng.random() < 0.5:
        return "noon", value

    style = rng.choice(["12h", "12h_space", "24h"])
    if style == "24h":
        return f"{hour}:{minute:02d}", value  # 14:30, 9:15

    hour12 = hour - 12 if hour > 12 else hour
    suffix = rng.choice(["am", "AM"]) if hour < 12 else rng.choice(["pm", "PM"])
    clock = f"{hour12}" if minute == 0 else f"{hour12}:{minute:02d}"
    space = " " if style == "12h_space" else ""
    return f"{clock}{space}{suffix}", value  # 3pm, 10:30 am


def random_attendees(rng):
    """1-3 names, written as 'Priya', 'Priya and Rahul' or 'Priya, Rahul and Ankit'."""
    people = rng.sample(NAMES, rng.randint(1, 3))
    if len(people) == 1:
        phrase = people[0]
    else:
        phrase = ", ".join(people[:-1]) + " and " + people[-1]
    return phrase, people


# ---------------------------------------------------------------------------
# Building examples
# ---------------------------------------------------------------------------
def make_example(rng, action, template):
    """Fill one template with random values and return the text and its label."""
    titles = REMINDER_TITLES if action == "set_reminder" else MEETING_TITLES
    label = {
        "action": action,
        "title": rng.choice(titles),
        "date": None,
        "time": None,
        "duration_minutes": None,
        "attendees": [],
    }
    slots = {"title": label["title"]}

    if "{date}" in template:
        slots["date"], label["date"] = random_date(rng)
    if "{time}" in template:
        slots["time"], label["time"] = random_time(rng)
    if "{duration}" in template:
        slots["duration"], label["duration_minutes"] = rng.choice(DURATIONS)
    if "{attendees}" in template:
        slots["attendees"], label["attendees"] = random_attendees(rng)

    return {"text": template.format(**slots), "output": label}


def build_split(rng, templates, n, used_texts, unseen_template):
    """Make n examples. `used_texts` is shared across splits so no text repeats
    anywhere (in particular, no test sentence also appears in training)."""
    examples = []
    while len(examples) < n:
        action = rng.choice(list(templates))       # actions are balanced
        template = rng.choice(templates[action])
        example = make_example(rng, action, template)
        if example["text"] in used_texts:
            continue
        used_texts.add(example["text"])
        example["template"] = template
        example["unseen_template"] = unseen_template
        examples.append(example)
    return examples


def save_jsonl(rows, path, prefix):
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            row = {"id": f"{prefix}-{i:03d}", **row}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    rng = random.Random(SEED)
    used_texts = set()

    train = build_split(rng, TRAIN_TEMPLATES, N_TRAIN, used_texts, unseen_template=False)
    test = (
        build_split(rng, TRAIN_TEMPLATES, N_TEST_SEEN, used_texts, unseen_template=False)
        + build_split(rng, TEST_ONLY_TEMPLATES, N_TEST_UNSEEN, used_texts, unseen_template=True)
    )
    rng.shuffle(test)

    OUT_DIR.mkdir(exist_ok=True)
    save_jsonl(train, OUT_DIR / "train.jsonl", "train")
    save_jsonl(test, OUT_DIR / "test.jsonl", "test")

    print(f"Reference date: {REFERENCE_DATE} ({WEEKDAYS[REFERENCE_DATE.weekday()]})")
    print(f"Wrote {len(train)} train examples and {len(test)} test examples to {OUT_DIR}")
    print(f"Test: {N_TEST_SEEN} with training templates, {N_TEST_UNSEEN} with unseen templates")


if __name__ == "__main__":
    main()
