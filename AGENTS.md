# AGENTS.md

## Source of truth
Read these first, in order:
- `README.md`
- `ARCHITECTURE.md`
- `docs/index.md`
- `docs/design-docs/*`
- `docs/product-specs/*`
- `docs/exec-plans/active/*`
- relevant code, tests, schemas, and generated artifacts

If docs and code disagree:
- trust the current code + tests for behavior
- update the docs in the same change

## Working rules
- Make the smallest change that satisfies the task.
- Prefer existing abstractions over new ones.
- Do not invent APIs, behaviors, or constraints.
- If something is unclear, inspect the repo before asking a human.
- Keep changes reversible.

## Planning
For non-trivial work:
- create or update a plan in `docs/exec-plans/active/`
- include: goal, scope, assumptions, steps, acceptance criteria, rollback notes
- keep the plan short and current
- The completed plan needs to be moved to `docs/exec-plans/done/`

## Execution
- Work from the task down to the smallest testable unit.
- Use repo-local docs, types, schemas, and tests as the guide.
- When blocked, identify the missing capability and encode it back into the repo.

## Verification
- Run the narrowest useful tests first.
- Add or update tests for every behavior change.
- Do not mark work done until acceptance criteria are met.
- If tests are flaky, record it and explain the impact.

## Review
Before opening or merging a PR:
- self-review the diff
- summarize behavior changes
- summarize verification
- list remaining risks and follow-ups

## Traceability
Every action MUST be logged.
- Every task have a uuid
- The item log is Structured
- Task, Actions, Observations, Result is MUST be include, Plan is Optional
- create log in `docs/logs/<task>-YYYY-MM-DD-uuid.md`

### Critic
MUST call another **subagent** to critically analyze the content of the log.

- Attach the critic report in `docs/logs/<task>-YYYY-MM-DD-uuid.md`.

- The critic should not be limited to a fixed structure. It should freely analyze:
  - what went wrong
  - what was missing
  - inefficiencies in the process
  - unexpected behaviors
  - incorrect assumptions
  - any other relevant observations

## Documentation hygiene
- Update or add docs in the same PR when behavior changes or when relevant documentation is missing.
- Prefer cross-links over duplication.
- Delete or deprecate stale docs.
- Keep generated artifacts clearly labeled.

## Documentation language and encoding

- All documentation must be provided in both English and Chinese and use UTF-8 encoding.
- English documentation should be placed alongside the relevant code or in the appropriate default location.
- Chinese documentation must be placed under the `docs/` directory with a `_zh` suffix.
- Each Chinese document should correspond to an English version, and their contents should remain consistent.

## Escalate to a human when
- operating environment is ambiguous
- product intent is ambiguous
- security, privacy, money, or data-loss risk is involved
- architecture decisions affect multiple subsystems
- the fix requires policy judgment outside the repo