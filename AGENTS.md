# AGENTS.md

## Source of truth
Read these first, in order:
- `README.md`
- `ARCHITECTURE.md`
- `docs/index.md`
- `docs/NAVIGATION.md`
- `docs/design-docs/*`
- `docs/product-specs/*`
- `docs/exec-plans/active/*`
- relevant code, tests, schemas, and generated artifacts

If docs and code disagree:
- trust the current code + tests for behavior
- update the docs in the same change

## Navigation-first rule
- `docs/NAVIGATION.md` and `docs/navigation-docs/*` define the active documentation routing and execution rules.
- `docs/NAVIGATION.md` defines **how the rule system executes**.
- `docs/navigation-docs/*` is the **content layer for module navigation and workstream status**.
- For task execution, follow the navigation graph and procedure described in `docs/NAVIGATION.md` before reading lower-priority docs.

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

## Coding Standards

### Readability — No Deep Nesting

Max 3 levels of indentation inside a function. Use early returns, guard clauses, or extract helpers.

```python
# BAD
def process(data):
    if data:
        for item in data:
            if item.is_valid():
                if item.price > 0:
                    do_something(item)

# GOOD
def process(data):
    if not data:
        return
    for item in data:
        _process_single(item)

def _process_single(item):
    if not item.is_valid() or item.price <= 0:
        return
    do_something(item)
```

### Determinism — Handle All Nulls and Edge Cases

Never assume inputs are valid. Every code path must be explicit.

```python
# BAD
def get_return_rate(current, previous):
    return (current - previous) / previous

# GOOD
def get_return_rate(current, previous):
    if previous is None or previous == 0:
        return None
    return (current - previous) / previous
```

### Naming — Precise, Logical, Consistent

- Functions: verb-first (`calculate_sharpe_ratio`, `fetch_daily_prices`)
- Variables: describe what it IS, not its type (`is_stock_selected` not `flag_bool`, `closing_prices` not `price_list`)
- Classes: noun, singular (`PortfolioOptimizer` not `OptHelper`)
- Keep naming style consistent across the entire module.

### Derivation First — Math Before Code

For complex financial, geometric, or physical logic, write the mathematical basis or pseudocode in an English comment BEFORE the implementation.

```python
# Sharpe Ratio: S = (E[R] - Rf) / std(R)
# where E[R] = mean of portfolio returns, Rf = risk-free rate
def calculate_sharpe_ratio(returns, risk_free_rate):
    ...
```