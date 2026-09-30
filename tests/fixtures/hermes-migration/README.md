# hermes-migration fixture (contract C7F-2)

This directory pins the Hermes host migration state of spec §9.9 (L1655, L1659): the file Hermes writes at
`<HERMES_HOME>/migrations/<provider>/<provider-epoch>/<import-run-id>.json` while it moves native
`MEMORY.md`/`USER.md` into an authoritative memory provider (contract C7F-1). Its top-level `state` is the
member ledger K-9 pins: `active`, `completed` or `rolled_back`; anything else, and an unreadable or corrupt
file, reads as active.

- `schema.json`: JSON Schema 2020-12 for the active manifest and the compacted receipt.
- `active.json`, `completed.json`, `rolled_back.json`: one example of each state, as RFC 8785 canonical bytes
  with no trailing newline and fixed timestamps (`created_at`/`updated_at`, Checkpoint A ruling R45-6).
- `SHA256SUMS`: the SHA-256 of exactly those four files.

Both the Hermes fork and the ygg repository hold `tests/fixtures/hermes-migration/`, and the four files listed
in `SHA256SUMS` are byte-identical in both. Every value is synthetic: no real secret, store, identity or path.
A change to any of them is a coordinated PR pair across the two repositories (spec §14.1 L2267).

Provenance: this is the Hermes fork's copy, written by `agent/memory_service/migration_manifest.py` and
checked by `tests/agent/memory_service/test_migration_manifest.py` (digests, a byte-exact round trip through
the loader, and the schema's member sets against the code's).
