# Recall: PRD

Status: draft for review. Nothing here is built. "Recall" is a working name.

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

### Should have

- Connections across subjects, such as a pre-meeting summary about one person
- Voice notes
- A daily or weekly digest
- A check-in when something important has gone quiet
- Entity matching, so "Mom" and "Amma" are recognized as the same person

### Could have

- Photos of receipts, tickets or posters
- Recommendations from personal history
- Pattern finding, such as sleep against coffee

## Architecture

```
Telegram  -->  Bot service  -->  Agent (Claude)
                                  tools: list_tables, create_table, add_row,
                                         update_row, query, search, set_reminder
                                    |
                     SQLite per user + embeddings
                                    |
                        Scheduler  -->  proactive messages
```

SQLite with one file per user keeps data separated and simple to export or delete.

## Key decisions

| Decision | Choice | Why | Trade-off |
| --- | --- | --- | --- |
| Interface | Telegram | Already on your phone, and capture is instant | Bot messages aren't end-to-end encrypted |
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

Targets marked "set after" need a first measurement, so that they aren't made up.

## Risks

| Risk | Impact | Plan |
| --- | --- | --- |
| Confident wrong answer | Loses trust at once | Cite the saved entry, and say when nothing was found |
| Messy schema | Answers get worse over time | Schema check, plus evals across long histories |
| Sensitive data | Privacy harm | Keep it local, offer delete, and warn users what not to send |
| Prompt injection in saved text | The agent could be manipulated | Treat stored text as data, and limit its tools |
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

The submission is still viable if work stops after step 5.

## Open questions

1. What's forgotten most often, and should that be the first use case?
2. Are English messages enough, or should other languages work too?
3. Should reminders ever fire on inferred promises, or only clear deadlines?
4. Should proactive messages have quiet hours by default?
