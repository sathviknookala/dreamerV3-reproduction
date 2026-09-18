# CLAUDE.md — dreamer

> **UNPOPULATED PROJECT.** The directory is empty as of 2026-09-17: no code, no commits, no
> artifacts. Every section below marked `TBD` is unfilled *on purpose* — nothing has been measured,
> so nothing is claimed. Fill each one the moment the corresponding fact exists, and delete this
> banner once the first four sections (statement, The System, Constraints, Core Hypothesis) are real.

**What this project is:** TBD — one paragraph: the domain, the unit under study, the language and
stack, what it is measured or compared against, and the scope discipline. It must answer what the
artifact is, what success looks like, and what the comparison point is. Also say what decides
priorities (e.g. "profiling decides what to optimize next, not a fixed plan").

## Doc map

No reference docs exist yet. This file is the always-loaded hub — when `docs/` is created, add one
line per doc here stating **when** to read it, and move the detail out of this file.

## The System

TBD — precise definition of the unit of work: inputs, outputs, shapes/types, stage order or control
flow, as a code block:

```
<input>  ->  <stage 1>  ->  <stage 2>  ->  <output>
```

Then one numbered item per stage with exact semantics — the formula, the tie-break, the cutoff rule,
the error case. This is the contract a correct implementation must meet. Where a step is
conventional but not actually required, say so and say what replaces it.

## Why It Is a Target

TBD — the cost model, corrected or confirmed by measurement, with the date in this heading. One
bullet per cost item: what it is, the measured share, the condition, and the artifact the number
came from. Then, explicitly, what the current path does *not* pay for — including any hypothesis of
your own that measurement killed, named with the number that killed it and the old framing quoted so
a reader can see it was retired on purpose.

## Target Regime / Constraints

TBD — the specialization *is* the project, so state it as fixed values or narrow ranges:

```
<Dimension>:   <fixed value or narrow range>
```

Then two to four sentences on what the narrowness buys: what a general implementation must handle
that this one may assume away, what that assumption unlocks, and what is explicitly not a goal.

## Core Hypothesis

> TBD — one falsifiable sentence, with the quantity and the comparison target named. If no
> measurement you could actually run would show it false, it is not a hypothesis.

**What it cannot claim.** TBD — the honest ceiling, with the number and artifact. State the
deliverable as what is actually shown.

**The bar is `<strongest real baseline>`, not `<the weak baseline>`.** TBD — name the strongest
available comparison, not the most flattering one.

## Workflow Rules

### Plan Before Acting
- Enter plan mode for any non-trivial task (3+ steps or architectural decisions)
- If something goes sideways mid-task, stop and re-plan — don't push through
- Write a spec or checklist upfront to reduce ambiguity; verify with the user before implementing

### Subagent Strategy
- Use subagents to keep the main context window clean
- Offload research, exploration, and parallel analysis to subagents
- One focused task per subagent

### Self-Correction Loop
- After any correction: note the pattern so the same mistake doesn't recur
- Ruthlessly iterate on this until mistake rate drops

### Verification Before Done
- Never consider a task complete without demonstrating it works
- Check logs, run tests, or diff behavior when relevant
- No build or test command exists yet. When one does, record it here verbatim and copy-pasteable.
  If the project grows a compiled or generated step, add the rule that a rebuild must precede
  testing — an unrebuilt edit tests the previous binary and passes — and declare inputs as
  dependencies in the build config so an edit triggers a rebuild
- A test that passes where the bug cannot occur is not a test — confirm it fails without its fix
- Record the domain-specific testing hazard here once known (nondeterminism, ties, ordering,
  floating point, clock/timezone, network flakiness): what a naive assertion would miss, and what to
  assert instead
- Ask: "Would a senior engineer approve this?"
- Before quoting a committed number, check the tree still reproduces it

### Demand Elegance
- For non-trivial changes: pause and ask "is there a more elegant solution?"
- If a fix feels hacky: "Knowing everything I know now, implement the clean version"
- Skip this for simple, obvious fixes — don't over-engineer

### Autonomous Bug Fixing
- Given a bug report: fix it; don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them

### Measurement Discipline
- Never quote a number that is not in a committed artifact under the results directory, and name the
  file. No results directory exists yet — create one before the first timed or scored run
- State what a metric excludes when it is a floor rather than a measurement
- Don't pipe a long run through `grep` — the pipeline reports grep's exit code and can turn a crash
  that lost real rows into an apparent success
- A run that must outlive the session has to be launched detached (`setsid`), not backgrounded
- Check the shared resource (GPU / port / DB / rate limit) is actually free before a timed run;
  another process's residency shows up as an error or as inflated timings, not as a clear message

## Code Style: Comments

- No paragraph-style or multi-line block comments explaining what code does
- Comments only where intent isn't obvious from the code itself (e.g. non-obvious tradeoffs,
  gotchas, why not what)
- Max 1 line per comment; keep it tight
- No section dividers, no docstrings restating the function signature, no "this function does X"
  fluff
- If the code is self-explanatory, leave it uncommented

## Code Style: language/stack

No code yet, so no conventions to record. Add this section only for conventions a session would
otherwise get wrong — naming, error handling, module layout, formatter/linter, forbidden idioms.
The code is the style guide; duplicating it here creates drift.

## Current Focus

**Empty project — nothing scaffolded, nothing measured, no stack chosen.** 0 commits, 0 tests, 0
artifacts. This `CLAUDE.md` is the only file.

Next: state what `dreamer` is and what it is measured against, and write that into the header
paragraph, The System, and Core Hypothesis. Until those three are real, no implementation work
should start. **Do not scaffold a directory tree, pick a framework, or write code before the
hypothesis and its baseline are named** — the layout is the cheapest thing to get right later and
the most expensive thing to have chosen for the wrong problem.

## Last Session

**Session 1 — bootstrap.** 0 commits.

- **Created this file from `~/agent_template.md`.** Verbatim sections carried over intact; every
  project-specific section left explicitly `TBD` rather than guessed.
- **Nothing measured, therefore nothing claimed.** No baseline, no cost model, no numbers anywhere
  in this file — by construction, not by omission.

## Known Issues

- **The project is not a git repository.** Nothing here is recoverable after an overwrite. Run
  `git init` and commit this file before doing real work.
- **This file is currently un-versioned and un-backed-up.** If it later becomes gitignored, anything
  that must survive belongs in a tracked doc under `docs/`, not here.
- **Every section marked `TBD` is an admission, not a placeholder to skip past.** A session that
  fills one in from plausible-sounding assumption rather than measurement has made the file worse
  than empty. Fill from artifacts or leave `TBD`.

---

At session end, refresh Current Focus / Last Session / Known Issues here — overwrite in place, 3–5
bullets in Last Session on what was actually done, no appending, no changelogs (git log is for
history). When reference docs under `docs/` are created, add a doc map section back to this file and
move the detail out of it — this file is the always-loaded hub and should stay thin.
