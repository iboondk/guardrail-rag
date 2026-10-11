# ADR-0002: Layered tenant isolation in one shared collection

- Status: Accepted
- Date: 2026-10-11

## Context

Several tenants share one Qdrant collection to keep operations simple. A single missed filter
would leak one tenant's documents to another. Filtering must hold even if a caller is careless,
and the claim must be testable.

## Decision

Isolation is enforced in four layers:

1. `TenantVectorStore` has no public search method that runs without a `tenant_id`. The
   `must: tenant_id == X` filter is built inside the store.
2. `validate_tenant_id` rejects anything outside `^[a-z0-9][a-z0-9_-]{1,63}$`, including empty
   strings, `*`, whitespace and injection-like input.
3. `HybridRetriever` checks every returned chunk. A mismatch raises `TenantIsolationError`
   instead of being filtered quietly, so the failure is visible.
4. Point IDs are `uuid5("tenant:doc:idx")`, so two tenants with the same `doc_id` never overwrite
   each other.

The leak suite plants a canary per tenant and runs queries from every other tenant. Any canary in
a retrieved chunk or answer counts as a leak.

## Consequences

- Isolation is checked by tests and by the eval runner, which exits non-zero on any leak.
- The check in layer 3 costs a loop over at most `top_k` chunks, which is negligible.
- Payload indexes (`is_tenant=True`) are a performance hint on server Qdrant. Local mode ignores
  them with a warning, so correctness must not depend on the index.
- Authentication is out of scope. A real system must take the tenant from an authenticated
  principal (docs/architecture.md section 1 records this limit).
