# Served-model capture from thinking-block signatures

Date: 2026-08-18. Status: approved by Jaedyn, implementing.

## Why

Every `thinking` block Claude Code stores carries a base64 `signature`. It is a
protobuf: top-level f1 = format version, f2 = envelope; the envelope's f1 is a
plaintext header whose f6 is the id of the model that actually produced the
block. Anthropic quietly serves some requests with a different model than the
one asked for (observed: `claude-carafe-416c93ba-v1-prod` under
`claude-opus-4-8` on 2026-07-22, `claude-kettle-e2c95a10-v2-prod` under
`claude-fable-5` from 2026-08-17). The header format has already changed four
times in six weeks and the v4 format (from 2026-08-19) omits f6, so this is a
best-effort observatory: capture the raw header (small, all the plaintext there
is), derive what we can server-side, and expect the signal to change or vanish.

Today the client ships the whole signature blob (mean 2.4 KB, 7% of all
uploaded bytes) and the server discards it. After this change the client
ships ~200 bytes of header and the server keeps it.

## Blob anatomy (measured on 20,852 blocks)

Top level: f1 varint version (absent, 2, or 4); f2 envelope; f3 varint always 1.
Envelope: f1 header (135 to 173 B); f2 12 B nonce; f3 12 B nonce; f4 48 B
wrapped key + tag; f5 ciphertext (the encrypted reasoning; length tracks the
hidden reasoning length).
Header: f1 varint (15 until 07-21, 16 since); f3 varint 2; f5 64 B per-block
digest; **f6 model id string**; f7 varint 0/1; f8 "thinking" | "narration";
f10 string tag (only `MYCRO_MODEL_MANATEE`, only on kettle blocks); f11 36 B
uuid constant; f14 16 B constant (from 08-18); f17 varint 1 (from 08-18).
v4 header carries only f1, f3, f7, f8.

## Wire contract (client -> server)

### Live ingest (`strip_content` in `client/claude-stats-push.py`)

For each `thinking` block that has a `signature`:

```
"signature": "[N chars]"            # blob dropped, same convention as thinking text
"sig_version": <int>                # top-level f1, 0 when absent
"sig_header": "<base64>"            # envelope f1 raw bytes, standard base64
"sig_cipher_len": <int>             # len(envelope f5), 0 when absent
```

If the blob cannot be parsed as (top-level protobuf with an f2 envelope whose
f1 is bytes): ship `"signature": "[N chars]"`, `"sig_error": true`, and
`"sig_sample": "<first 256 chars of the base64>"` so a schema change leaves a
diagnosable sample. Never raise; never drop the event.

The client does NOT decode f6. One decoder, server-side.

### Backfill (`client/backfill-transcripts.py` -> `POST /api/backfill`)

New field on `BackfillRequest`:

```
"sig_headers": { "<event uuid>": [<sig_version>, "<sig_header b64>", <sig_cipher_len>], ... }
```

Same 20,000 cap and batching as the other maps. Fill-only-unset: the server
updates only rows where `sig_header IS NULL`. Bump `CACHE_VERSION` in the
backfill client so machines that ran an older backfill rescan.

The event uuid is the transcript record uuid (`rec["uuid"]`), one thinking
block per assistant record in Claude Code transcripts (each content block is
its own record). If a record ever has several thinking blocks, use the first.

## Server

### Schema (`app/db.py`, additive, via the existing ADD COLUMN helper)

```
served_model    TEXT      -- header f6, NULL when absent
sig_version     INTEGER   -- top-level f1
sig_header      TEXT      -- base64 raw header
sig_cipher_len  INTEGER
sig_fields      TEXT      -- header field numbers present, e.g. "1,3,5,6,7,8,11"
```
Partial index: `CREATE INDEX IF NOT EXISTS idx_events_served ON events(day, model, served_model) WHERE sig_header IS NOT NULL`.

### Decoder (`app/sigheader.py`, pure stdlib, tolerant)

`decode_header(b64) -> {"served_model": str|None, "fields": "1,3,...", "kind": str|None, "tag": str|None}`;
never raises on garbage (returns Nones and empty fields). Also
`split_signature(b64) -> (version, header_b64, cipher_len)` for the fallback
below and for tests; the client carries an identical copy of `split_signature`
(a test asserts the two function sources match, to stop drift).

### Ingest (`_extract_event`, thinking branch)

- If the block carries `sig_header`: store it plus `sig_version`,
  `sig_cipher_len`; derive `served_model` and `sig_fields` with the decoder.
- Else if the block still carries a raw `signature` (older client): run
  `split_signature` server-side and proceed as above. Old clients keep working.
- Coerce/validate: `sig_header` must be base64 <= 4096 chars; ints bounded;
  strings truncated (served_model <= 64, sig_fields <= 128).

### Backfill (`backfill()`)

Loop `req.sig_headers`: validate as above, `SELECT day FROM events WHERE uuid=?
AND sig_header IS NULL`, `UPDATE events SET served_model=?, sig_version=?,
sig_header=?, sig_cipher_len=?, sig_fields=? WHERE uuid=?`. Count as
`updated_sig_headers` in the response. Touched days feed the existing reroll
set only if a rollup depends on them (see below; if the rollup is on-the-fly,
do not reroll for this).

### API

`GET /api/served-models?days=30` (dashboard-auth, personal scope) ->
```
{"days": 30, "rows": [
  {"day": "2026-08-17", "model": "claude-fable-5", "served_model": "claude-kettle-e2c95a10-v2-prod",
   "sig_version": 2, "sig_fields": "1,3,5,6,7,8,10,11", "blocks": 189, "cipher_bytes": 123456}, ...]}
```
Grouped by (day, model, served_model, sig_version, sig_fields); `served_model`
null rows included (they are the "hidden" v4 share). Computed on the fly with
the partial index; no new rollup table.

### Dashboard

One small dim chip on each model row/card when, in the selected range, any
block for that model has `served_model` set and different from `model`:
`58% kettle-e2c95a10-v2` (slug with `claude-` and `-prod` trimmed; multiple
slugs joined by " · ", most common first). Nothing when there is nothing to
report; nothing for null served_model. Keep it visually deprioritized (the
existing muted text style), never in the model name itself.

## Tests

Server: decoder on real captured headers (fixtures: fable v2, kettle v2 with
f10, opus-4-8 v0, fable v4 without f6, garbage), ingest with client-decoded
fields, ingest fallback with raw signature, backfill fill-only-unset, API
grouping, chip rendering present/absent. Client: strip_content emits the new
fields and drops the blob, error path emits sig_error + sig_sample, backfill
harvest emits sig_headers, cache version bump forces rescan.

## Out of scope here

dotfleet vendoring of the client, the one-shot fleet backfill, and the
statusline (already done in dotfleet). Jaedyn does the deploy after local E2E.
