# Session parent: opt-in implementation and release gates

## Scope and defaults

- Base: develop `9708953bef9cff32fcd9af499b7cdcfa3b1bc67c`.
- `MORMI_SESSION_PARENT_GRAPH_ENABLED=false` (emergency turn-only bypass).
- `MORMI_SESSION_PARENT_GRAPH_CANARY_PERCENT=0` (NEW V2 enrollment only).
- `MORMI_SESSION_PARENT_STORE_TIMEOUT_SECONDS=0.5` bounds optional cursor I/O
  only. Model timeouts and the canonical domain commit are unchanged.
- No change to `MORMI_RUNTIME_CONTRACT_VERSION` or the V2 canary percentage.
- Existing non-enrolled conversations and legacy sessions stay turn-only. Merely
  raising the percentage does not migrate them. Existing enrolled conversations
  continue at percentage zero unless the enabled flag is switched off.
- No API request/response/SSE changes, no BE/FE edits, no model or prompt changes.

## Persistence and resume contract

The parent graph has `START -> wait_for_input -> execute_turn -> wait_for_input/END`.
Creation keeps the current initial-turn transaction; only the committed winner
is enrolled. At the first response, the parent bootstraps the initial WAIT from
the canonical DB. Later requests restore its WAIT checkpoint and resume using a
response-ID token, never child text. No process or HTTP connection stays alive
while waiting for a child. No child input means no model or pedagogical work.

An invocation-local staging saver backs the graph. Its latest WAIT/END packet
is durably stored in the same application database, not in process memory alone.
The rest of the graph's intermediate writes are discarded after the request.
This is deliberately **turn-boundary recovery**, not durable recovery in the
middle of an LLM call. It does not add autonomous retries or time-travel APIs.

The only new table is `dialogue_session_parents`. It holds the pinned workflow
version, committed DB version, current turn, phase, generation, and a bounded
JSON checkpoint. No `SessionState`, raw utterance, full result, model request,
history, arbitrary Python object, or exception is persisted in that packet.
The native child and retry graphs explicitly disable checkpoint inheritance.
Tracing stays disabled for request content.

The existing service remains responsible for its canonical commit and existing
post-commit profile update. Parent END does not repeat either. In particular:

- A task-complete branch may transition primary -> transfer while the session
  stays ACTIVE. Only the committed session COMPLETED state ends the parent.
- An earlier response ID replays its ORIGINAL result, not the latest cursor.
- Existing concurrent-response losers can be stale/409; this feature does not
  promise to deduplicate simultaneous paid model calls.
- Domain commit failure leaves the old DB turn authoritative. A later explicit
  client retry may execute it; no unattended background retry is added.
- After domain success, cursor-write/projection failure cannot turn the result
  into a new child-facing error or rerun the completed turn. The next new input
  rebuilds a WAIT checkpoint from the DB. Replay itself remains side-effect-free.
- Publication holds a short conversation-row lock, checks the canonical version,
  then compares cursor generation. No DB lock is held during model execution.
- Cursor absence/read failure degrades to the original turn-only path. Recovery
  never deletes domain facts or treats uncommitted graph state as learned facts.
- Cursor load/enroll/publish timeout cancels only that optional operation. A
  broken bootstrap can use the existing service ONLY before any domain work has
  started. After domain work starts, existing errors propagate without new
  retries; after a final committed result exists, cursor projection failures
  return that result without re-execution. This is an execution-path bypass,
  not a change to educational fallback or speaker routing.

## Migration and rollback

Revision `20260831_07` adds only the optional cursor table. The migration runner
supports both the old identity targets (05/06) and the new head/07. New DB stamping
and schema validation cover the added table. Turn-only startup still accepts the
06 domain schema; enabling the parent additionally requires a complete cursor
schema. Domain columns, observation/outbox formats and ladder model inputs are
unchanged.

1. Deploy/migrate with the parent disabled.
2. Verify all release gates below against the exact candidate.
3. Enable only test-account/new-conversation traffic, then a small percentage.
4. Monitor parent rebuild/deferred publication, child errors, model calls,
   duplicate notes/outbox, completion facts, latency and DB usage before expansion.

Emergency rollback: set `MORMI_SESSION_PARENT_GRAPH_ENABLED=false` in the SAME
compatible image. Enrolled conversations resume through the old turn-only
service using the canonical DB. Re-enabling repairs a stale parent cursor.
Do not equate this with reverting to V1 or blindly rolling back to an image
whose migration runner does not know revision 07. No down migration is needed
for the feature bypass.

## Validation

Run without paid providers:

```sh
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m ruff check src tests scripts
PYTHONPATH=src .venv/bin/python -m mypy src/mormi_api scripts
PYTHONPATH=src:tests .venv/bin/python tests/benchmark_session_parent.py
```

The frozen SERVICE reference is independent of the new service routing code.
The prior frozen turn-engine reference remains unchanged. Tests compare model
requests/call order, full envelopes/progress, all pre-existing DB tables, note
provenance and profile/outbox results for home 9, cafe 4 (including compatibility),
and amusement 3 scenarios, direct/choice/joint paths, fresh instances and bypass.
Fault tests cover stream close/cancel, storage failure, corrupted/stale cursors,
stale input, replay, checkpoint privacy and bounded checkpoint size.

The PostgreSQL multi-process release gate is NOT satisfied by SQLite tests.
It is intentionally skipped unless `MORMI_TEST_POSTGRES_URL` explicitly points
to a LOCAL dedicated `mormi_test_*` database using the asyncpg driver. It creates
and removes only a unique synthetic schema in that database. It never reads
the application's `.env` database setting.

```sh
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_session_parent_postgres.py
```

**Current release status:** local synthetic regression/migration/fault checks
and actual PostgreSQL 16.15 multi-process checks have passed. Deployment/migration
and live-model smoke remain required before production activation. No production
changes or paid model calls were made for this implementation.

The PostgreSQL gate runs duplicate response IDs and competing distinct response
IDs three times each (6 checks), using two OS processes and separate connections.
Each run verifies one winning commit, one stale loser, one observation/note,
terminal parent state and rejection of an outdated cursor writer. It uses a
temporary localhost-only container, no existing service containers or volumes.
The initial test-harness failure was a transactional `SET search_path` being
rolled back during connection setup; using asyncpg startup `server_settings`
fixed test isolation without modifying application code. Passing these checks
does not claim model-call deduplication or a production load-test result.

Final local run with the dedicated PostgreSQL URL (2026-08-31):
**1186 passed, 0 skipped**, 14 existing deprecation warnings (76.96 seconds).
Ruff and mypy (application plus scripts, 69 files) also passed.
The task-owned temporary container and its in-memory synthetic database were
removed after verification; existing containers, images and volumes were retained.

Initial synthetic SQLite service benchmark (80 measured responses per executor,
no model calls): baseline median 9.644ms / p95 10.233ms; parent median 14.070ms /
p95 19.842ms (+9.609ms). This is a local overhead measurement, not production
throughput or a proof that deployment performance is acceptable.
