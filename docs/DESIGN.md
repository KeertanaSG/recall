# Recall: Design

Why Recall is built the way it is: the problems that shaped it, the options considered, and the known
weak spots. For what the code looks like, see [ARCHITECTURE.md](ARCHITECTURE.md). For what the
product should do, see [PRD.md](PRD.md).

## What shaped the design

Recall is a personal memory, so a few constraints decide most of the design:

- **One confident wrong answer breaks trust.** When goals conflict, accuracy wins over speed and cost.
- **Capture must take seconds.** Asking people to confirm or fill in forms would kill the habit.
- **It can hold any subject.** There's no fixed list of categories, and no rule may be written for
  one subject or example.
- **Saved text is untrusted.** Anything in memory could contain instructions, including text pasted
  from elsewhere.
- **Cost must stay near zero per message.** The target is under 0.1 cents.

## Structured memory, not a pile of notes

**Decision:** the agent writes typed rows into tables it designs, and also embeds each row for search.

| Option | Exact totals | Vague recall | Any subject | Verdict |
| --- | --- | --- | --- | --- |
| Vector store only | No | Yes | Yes | The model has to add up numbers it reads, and LLMs are unreliable at totals |
| Fixed schema | Yes | Partly | No | Every new subject needs code |
| Agent-designed tables plus embeddings | Yes | Yes | Yes | Chosen |

The cost is that tables can get messy. Three things limit that:

1. **Opus designs tables, and Haiku does everything else.** A table outlives the message that created
   it, so a better design pays off many times. Most saves never reach this step.
2. **A duplicate check runs before each new table.** It compares names, column overlap and description
   similarity. The agent can only override a near match by confirming it's a different subject.
3. **New details become columns.** `add_column` lets a table grow without being redesigned.

Coercing values to their column types matters as much. It's what makes `SUM(amount)` trustworthy.

## Answers you can check

**Decision:** the model cites rows, and Recall replaces those citations with rows from the database.

A model can invent a source as easily as a fact, so a source it writes proves nothing. Recall only
takes a `[table#id]` from the model, then looks the row up. If the row doesn't exist, the reply says
so. The same idea applies to nudges: keys the judge makes up are dropped, so a nudge always points at
a real entry.

## Retrieval

**Decision:** hybrid search, then a relevance grade, with a free local search first.

Plain vector search was the baseline, and on the benchmark it answered 21 of 24 answerable questions
and got 0 of 10 nothing-saved questions right. It always returns something, so it can't tell you
that nothing matches. Each addition solved a specific problem:

| Addition | Problem it solves |
| --- | --- |
| Keyword search | Embeddings blur names and exact words |
| Reciprocal rank fusion | Semantic and keyword scores can't be compared, but their ranks can |
| Query planning | A question and its answer can share no words |
| Relevance grading | Lets Recall say it found nothing |
| Local search first | Simple questions need fewer model steps |

Search-first took the median answer from about 4.8 seconds to between 2.8 and 3.9, depending on API
load. It scored 24/24 and 9/10.

A retry step used to follow grading, but evals showed a second attempt added time without fixing
anything, so it was removed. The pipeline stays in LangGraph, which keeps its steps separate and
makes it easy to add a step back.

## Storage

**Decision:** one SQLite file per user, with sqlite-vec for vectors.

| Option | For | Against |
| --- | --- | --- |
| One SQLite file per user | Free, strong isolation, simple export and delete, no server | One machine only |
| Hosted Postgres with pgvector | Many servers, managed backups | Costs money, and a query bug could cross users |

Isolation decided it. With separate files, a bad query can't reach another user's rows, however it's
written. `MemoryStore` and `StoreProvider` keep a later move to Postgres contained, and the plan is to
move only when Recall outgrows one server.

**Encryption:** Recall relies on disk encryption rather than SQLCipher for now. It covers the main
risk, a stolen disk, with no key management. SQLCipher stays on the roadmap once it's tested with
sqlite-vec.

## Trust boundaries

Security rests on limiting what a compromised model can do, not on assuming it won't be compromised.

| Threat | Defence |
| --- | --- |
| A saved note tells the agent to delete everything | The agent has no delete tool. Deleting takes a tap from the user |
| A saved note tells the agent to run harmful SQL | SQL runs on a separate read-only connection with an authorizer, a time limit and a row cap |
| A leaked write token reads memory through the agent | `remember` is add-only, and returns only new entry IDs |
| A token reaches full chat | Chat needs `OWNER`, a scope tokens can never hold |
| A caller asks for another user's data | No service method takes a user ID |
| A stolen token store | Only SHA-256 hashes are kept |
| Spending runs away | Daily limits per user and in total, across processes |
| Logs leak memory | Logs carry IDs and counts, and exception messages are dropped |
| Saved text acts as a template | Values are inserted as plain text, never as template code |

The injection eval tried 9 attacks on real Claude, and all were resisted with no data changed. That's
one run of a small set, so it shows the defences work, not that they're unbreakable.

## Proactive messages

**Decision:** a free detector, then a cheap judge, then the normal reminder rules.

Asking a model to read everything each day would be expensive and noisy. So detection uses only the
schema and local search: yes/no columns like `repaid`, plus searches for kinds of unfinished business
like "someone owes me money". Those phrases describe kinds of open loops, not subjects, so they fit
whatever tables a user has.

Haiku then sees only those candidates, along with later entries that might show they're resolved.
Precision matters more than recall here, because every false alarm teaches someone to mute Recall.
The labelled eval scored 100% precision, with 6 of 7 open items found.

Picks become ordinary reminders, so there's only one set of delivery rules to get right.

## Model choices

**Decision:** Haiku for most steps, Opus for table design, local embeddings.

| Option | Result |
| --- | --- |
| Haiku for the agent and search | Meets the accuracy and cost targets, at about 0.06 cents per question |
| Fast mode | Rejected on cost |
| Local LLM (Qwen3) | Stalled for hours on the benchmark |
| LiteLLM | Matched accuracy, but answered about 0.9 seconds slower, and its PyPI package was compromised twice in 2026 |
| Local embeddings | No API cost, and note text isn't sent to an embedding provider |

LiteLLM's goal, swappable providers, is met by the `LLM` interface instead. It costs one small class
per provider and adds no dependency to the path every message takes.

## Modularity

**Decision:** interfaces only where swapping is realistic.

The LLM, embedder, storage and channels sit behind interfaces, because each could plausibly change.
A second provider with a different message format runs in the tests to prove the seam holds. Other
parts stay concrete, because an interface with no realistic second version is only extra code.

The code also stays deliberately small. Modules were merged where they were thin, prompts moved out
into templates, and tests are table-driven. `storage/store.py` is the one large file, at about 1,000
lines, and is the first candidate to split.

## Known weak spots

- **Evals are synthetic and small.** 34 retrieval questions and 24 nudge rows are early checks, and
  real use may show different results.
- **Some thresholds are rough.** The table similarity limits are uncalibrated, and `MIN_MATCH_SCORE`
  was set from the demo data.
- **New tables are slow.** A save that designs a table can take 10 to 20 seconds.
- **Reminders depend on the bot process.** If the machine is off, nothing sends until it's back.
- **Each process loads its own embedding model,** which adds memory use.
- **Vector size is fixed at 384,** so an embedder with a different size needs a re-embed.
- **Prompts are tuned for the current models,** so a new provider has to pass the evals.
- **Rate limits are per token,** so a user with several tokens gets the limit on each one.
- **Message text passes through Telegram and Anthropic,** and bot chats aren't end-to-end encrypted.
