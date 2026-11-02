# Recall

Text it anything. It remembers, organizes it, and reminds you before you forget.

Recall is a Telegram bot that turns one-line messages into structured personal records. Tell it
"Cab to airport 450" and it saves a row. Ask "How much did I spend on cabs this month?" and SQL
does the math, with the answer citing the rows behind it.

The product thinking is in [docs/PRD.md](docs/PRD.md).

## How it works

```
Telegram --> Bot (bot.py) --> Agent (agent.py) --> Tools (tools.py)
                  |                                      |
           Reminder tick                     Per-user SQLite + sqlite-vec (store.py)
```

- **Agent-made tables:** Claude chooses an existing table or creates one. `create_table` refuses
  names that mean the same thing (`coffee`, `coffees`, `coffee_log`) and asks for confirmation when
  the columns or description look similar to an existing table.
- **Designed schemas:** a new table is designed by Claude Opus in `schema_design.py`, so it suits a
  whole subject. Required columns are enforced with `NOT NULL`, everything else is optional, and new
  details arrive through `add_column`. Everything else runs on Claude Haiku to stay cheap.
- **Exact and fuzzy retrieval:** totals and filters go through SQL. Vague questions go through a
  LangGraph RAG pipeline in `retrieval.py`: plan searches, merge semantic and keyword matches with
  reciprocal rank fusion, then grade them for relevance. Rows and vectors live in the same file.
  `evals/run_retrieval.py` compares this with plain vector search on a 5,500-row history.
- **Answers you can check:** the model cites rows as `[table#id]`. The bot looks each one up and
  prints the stored row, so sources come from the database, not the model. A citation that doesn't
  resolve shows a warning.
- **History and undo:** every write is logged under its turn. Corrections keep old values, undo
  reverses a whole turn, and a turn that fails midway rolls back.
- **Forget means gone:** `/forget` removes the row, its history, linked reminders and recent chat
  context, and overwrites the deleted bytes.
- **Limited tools:** the agent has no delete tool, and its queries run on a separate read-only
  connection that can't see internal tables. Deleting always takes a tap from the user, so text
  stored in a row can't trigger it.
- **Reminders:** requested reminders fire on time. Proactive ones respect quiet hours, pause, muted
  types and a daily limit, and held ones go out at the next allowed moment.

## Setup

You need [uv](https://docs.astral.sh/uv/), a bot token from [@BotFather](https://t.me/BotFather), an
Anthropic API key, and your numeric Telegram user ID (ask
[@userinfobot](https://t.me/userinfobot)). Embeddings run locally, and the small model downloads to
`data/models/` on first start.

```sh
cp .env.example .env   # then fill it in
uv sync
uv run recall
```

Run the tests with `uv run pytest`. They use no network and no API keys.

## Deploying

Run one process on a small always-on server, because reminders need the scheduler to stay up.
[deploy/recall.service](deploy/recall.service) is a systemd unit that expects the project in
`/opt/recall` and a `recall` user. `.env` should be readable only by that user. Back up `data/`,
which holds every user's database.

## Commands

| Command | What it does |
| --- | --- |
| `/undo` | Reverse the last save |
| `/forget <words>` | Find an entry and erase it for good, after a tap |
| `/export` | Download everything as CSV files in a zip |
| `/delete_all` | Erase everything, after a tap |
| `/timezone <Area/City>` | Set your time zone |
| `/quiet 22:00 08:00`, `/quiet off` | Set quiet hours |
| `/pause`, `/resume` | Hold or restart proactive messages |
| `/unmute <kind>` | Turn a reminder type back on |

## Privacy

This is a test project. Data is stored unencrypted on your server in one file per user, with
owner-only permissions. Message text also passes through Telegram and Anthropic. Telegram bot chats
aren't end-to-end encrypted.

## Access and limits

`ALLOWED_USER_IDS` is either a list of Telegram IDs or `*` for anyone. Every user gets their own
database, whichever you choose. Daily message limits cap API spending, at 30 per user and 300 in
total by default. Commands don't count, because they don't call Claude. Set a spend limit in the
Anthropic Console as well.
