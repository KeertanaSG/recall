# Changelog

Notable changes to Recall, newest first, grouped by pull request. There are no tagged releases yet,
so everything is version 0.1.0.

## 2026-11-10

### Ranked keyword search

- Keyword search now uses an SQLite FTS5 index ranked by BM25, instead of `LIKE` with newest rows
  first. All keywords go in one query, so a row matching several of them ranks first, and a busy
  table no longer crowds out the others.
- The index uses the trigram tokenizer, so a term still matches inside a word, as `LIKE` did.
  Terms under three characters, like `Al`, are matched by a scan and listed after ranked results.
- The index updates in the same transaction as each write. Existing databases build it on first
  open, and `forget` merges it so erased words don't stay in it.
- With no model calls on the 5,524-row eval history, the same 22 of 24 needles make the top 15
  and mean reciprocal rank rises from 0.612 to 0.623. One question moved from 13th to 4th.

## 2026-11-02

### Every-feature eval

- Added `evals/run_messages.py` and a dataset of 430 cases covering every feature. The latest run
  passed 427.

### MCP from anywhere (#13)

- MCP over HTTP now accepts `RECALL_MCP_PUBLIC_URL` as a host, so it can run behind a tunnel. All
  other hosts and origins are refused.
- Added setup steps for Claude Code and for a Cloudflare quick tunnel.

### Documentation (#11, #10)

- Added `requirements.txt` with pinned, hashed versions, so Recall installs with pip as well as `uv`.
- Rewrote the README's quick start as step-by-step setup.
- Brought the PRD and roadmap up to date with what's built.

### Testing and verification (#8)

- Reached 100% line and branch coverage with table-driven tests. Coverage below 100% now fails the
  test run.
- Added a concurrency test using real processes.
- Added a prompt injection eval. All 9 attacks were resisted in one run on real Claude.
- Corrected the recorded answer speed to a range of 2.8 to 3.9 seconds, depending on API load.
- Recorded the decision to stay on the Claude SDK. The LiteLLM trial in #9 was closed without merging.

### Smaller codebase (#7)

- Grouped templates and dropped thin modules, with no features removed.

### Swappable parts and simpler code (#6)

- Split the code into `core`, `llm`, `storage` and `channels` behind interfaces.
- Added tests that enforce the module boundaries and run a second, fake provider.
- Moved every prompt into Jinja templates.
- Added structured JSON logging across all three processes.
- Added failure handling at the points where failures cause real harm.
- Fixed several failures that could mislead users:
  - A crash no longer leaves a partly saved turn.
  - A chat history failure no longer turns a good save into an error.
  - A failed `remember` no longer looks like "nothing to save".
  - One user's failure no longer stops other users' reminders.
- Removed fast mode, which was rejected on cost, and the search retry loop, which added time without
  fixing anything.

### Person briefs and search-first (#5)

- Added person briefs through a new `about` tool.
- Added `recall-demo`, which seeds 36 backdated entries with no model calls.
- Turned on search-first by default, cutting the median answer from about 4.8 seconds.
- Added paging to `/forget` and a status message while reading a photo.
- Saved bot logs to a rotating file.
- Shortened long source lists and fixed citations leaking in mixed-table answers.
- Fixed person briefs reversing who owed whom.
- Stopped crisis replies giving phone numbers the model can't check.

### README (#4)

- Rewrote the README for new readers.

### Smart nudges and photo capture (#2, #3)

- Added the detect, judge, policy and deliver nudge pipeline, plus `/nudge`.
- Added photo capture, with receipt items saved as separate rows. Photos are never stored.
- Added a labelled nudge eval with saved results.

### Secure gateway and first version (#1)

- Built the first version:
  - Per-user SQLite storage with history, undo and forget
  - Agent-designed tables with a duplicate check
  - Exact answers through read-only SQL
  - LangGraph search with relevance grading
  - Citations read from the database
  - Reminders with quiet hours and daily limits
  - The Telegram bot and its commands
  - A retrieval benchmark
- Added `RecallService` as the one path to a user's memory.
- Added hashed API tokens with scopes, expiry and an audit log.
- Added the HTTP API and the MCP server.
- Made daily limits hold across processes.
- Made `remember` add-only, so a write token can't read memory through the agent.
