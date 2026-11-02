# Recall: PRD

Status: core product built and running locally. Last updated 2026-11-02. See [ROADMAP.md](ROADMAP.md) for what exists, how it was tested and what comes next. "Recall" is a working name.

## One-liner

Text it anything. It remembers, organizes it, and reminds you before you forget.

## Problem

People forget constantly: what they watched, who they owe, what they promised, and what they planned. The tools that help are split across notes, calendars, reminders and spreadsheets, and each one needs effort at the exact moment you're busy.

So most things never get saved. When they are saved, they sit in separate places that can't be counted or connected. A reminder app only helps if you remember to set the reminder.

## Target user

**Primary:** a busy person who forgets everyday details and won't maintain a system. The first user is the builder.

**Moment of use:** something happens, and they have five seconds to capture it before moving on.

## Alternatives

| Alternative | Where it falls short |
| --- | --- |
| Reminder and calendar apps | You must set them up yourself, and they only hold dates and tasks |
| Notes apps | Easy to save to, but hard to count, query or connect |
| Spreadsheets and trackers | Exact, but need setup and discipline, usually one per topic |
| Chat assistants with memory | Good at remembering facts, but not built as queryable personal records you can total, inspect and export |

## Positioning

Recall is not trying to remember better than an AI assistant. Its difference is the combination:

1. **Effortless capture:** text one line, with no forms or setup.
2. **Structured records:** every memory can be counted, filtered, corrected and exported.
3. **Proactive:** it speaks up before you forget, without being asked.
4. **Connected:** it links movies, money, people and plans into one view of your life.
5. **Inspectable:** every answer shows the saved entry behind it.

Large assistants could add these features. The durable value is trust and the user's own accumulated record.

## Goals

1. Saving a memory takes under 10 seconds.
2. Recall answers questions about saved information correctly and shows its source.
3. It sends useful reminders the user never explicitly set up.
4. A user keeps using it for a week.

## Non-goals

- A new messaging app. Telegram is the interface.
- Passive recording of calls, screens or conversations
- Storing passwords, bank details or medical records
- Group use or shared memories
- Medical, legal or financial advice

## User flows

### Capture
The user texts "Watched Dune 2 with Arjun, 9/10." Recall saves it, confirms in one line, and offers an undo.

### Recall
The user asks "What sci-fi movie did I love last year?" Recall answers and shows the saved entry.

### Analyze
The user asks "How much did I spend on cabs this month?" Recall runs an exact query, so the answer is calculated, not estimated by the model.

### Proactive
Recall texts "You told Priya you'd send the file by Friday." The user can mark it done, snooze it, or stop that type of reminder.

### Correct
The user says "I moved to Pune." Recall updates the record and keeps the old value as history.

### Control
The user can run `/export`, `/forget` and `/delete_all`, and can pause proactive messages.

## Requirements

### Must have

| Area | Requirement |
| --- | --- |
| Capture | Accept natural text in Telegram, for any subject |
| Organization | An agent chooses an existing table or creates a new one when needed |
| Schema control | Before creating a table, the agent checks for a similar one, so `coffee` and `coffees` don't both appear |
| Retrieval | Exact questions use SQL, and fuzzy questions use semantic search |
| Reliability | Answers cite saved entries. If nothing is found, it says so instead of guessing. |
| Time | Understands dates like "next Friday" in the user's time zone |
| Proactive | Detects deadlines and promises, then sends reminders on time |
| Updates | Handles corrections and changed facts, and keeps history |
| Control | Undo, export, forget and delete everything |
| Safety | Read queries are read-only and limited to that user's data. Deleting needs confirmation. |

All of these are built.

### Should have

- Connections across subjects, such as a pre-meeting summary about one person (built as person briefs)
- A check-in when something important has gone quiet (built as smart nudges)
- Voice notes (not built)
- A daily or weekly digest (not built)
- Entity matching, so "Mom" and "Amma" are recognized as the same person (not built)

### Could have

- Photos of receipts, tickets or posters (built)
- Recommendations from personal history (not built)
- Pattern finding, such as sleep against coffee (not built)

## Architecture

```
Telegram   HTTP API   MCP server
    \          |         /
          RecallService  (access checks, limits, audit)
                |
    Agent, tools, retrieval, nudges
          |                 |
      LLM interface     Storage interfaces
      (Claude)          (SQLite per user, sqlite-vec, local embeddings)

Telegram scheduler  -->  reminders and nudges
```

SQLite with one file per user keeps data separated and simple to export or delete. The LLM, embedder and storage sit behind interfaces chosen in `factory.py`, so each can be replaced without touching the core.

## Key decisions

| Decision | Choice | Why | Trade-off |
| --- | --- | --- | --- |
| Interface | Telegram | Already on your phone, and capture is instant | Bot messages aren't end-to-end encrypted |
| Other apps | HTTP API and MCP, with scoped tokens | Lets tools like Claude Code read and add memories | More to secure, so neither offers a delete |
| Storage | Tables plus embeddings | Tables give exact answers, and embeddings find vague memories | Two systems to keep in sync |
| Schema | Agent-created, with guardrails | Works for anything, with no setup | Risk of messy tables, so this needs evals |
| Math | SQL does the counting | LLMs are unreliable at totals | Needs safe query tools |
| Writes | Auto-save with undo | Confirmation on every message would kill the habit | Some saves will be wrong, so show what was saved |
| Reminders | Only high-confidence ones, with limits | Too many nudges make people mute it | It'll miss a few |
| Data | Local for the POC | Personal data needs caution | No hosted multi-user product yet |

## Success metrics

**Primary:** correct answers on a scripted test of real-life conversations, compared with a plain vector-search baseline.

| Type | Metric | Target |
| --- | --- | --- |
| Capture | Facts extracted correctly | Set after the first eval run |
| Schema | Duplicate or misplaced tables | Set after the first eval run |
| Recall | Correct answers with the right source | Higher than the baseline |
| Reminders | Reminders the user finds useful | Most of them, rated by testers |
| Habit | Days used in the first week | At least 4 of 7, for 2 or 3 testers |

Targets marked "set after" need a first measurement, so that they aren't made up. Capture and schema evals are still to come. Retrieval, nudge and injection results so far are in the [README](../README.md) and [ROADMAP.md](ROADMAP.md).

## Risks

| Risk | Impact | Plan |
| --- | --- | --- |
| Confident wrong answer | Loses trust at once | Cite the saved entry, and say when nothing was found |
| Messy schema | Answers get worse over time | Schema check, plus evals across long histories |
| Sensitive data | Privacy harm | Keep it local, offer delete, and warn users what not to send |
| Prompt injection in saved text | The agent could be manipulated | Treat stored text as data, and limit its tools. All 9 attacks were resisted in one run on real Claude |
| Too many reminders | User mutes it | Daily limit, and easy feedback |
| Forgetting to use it | No data, no value | A gentle digest, and extremely fast capture |
| Competition from large assistants | Weakens the pitch | Focus on structured records, proactivity and inspectability |

## Build order

1. Telegram bot with capture and undo
2. Agent with dynamic tables and schema control
3. Exact and semantic retrieval with citations
4. Reminders and deadlines
5. Eval suite with long simulated histories
6. Updates, corrections, export and delete
7. Connections across subjects, and the digest

All of these are built except the digest. Step 5 is partly done: there's a 5,524-row retrieval benchmark, but not yet the subject-by-subject evals.

## Decided

- **Languages:** English only for now.
- **Inferred promises:** proactive messages can fire on them. A nudge must pass a Haiku judge and the reminder rules, and each one has a stop button.
- **Quiet hours:** on by default, from 22:00 to 08:00. Reminders the user asked for still fire on time.

## Open questions

1. What's forgotten most often, and should that be the first use case?
