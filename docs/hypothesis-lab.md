# Hypothesis Lab: adding a speculative idea

The Hypothesis Lab is a collection group for ideas that are **not established physics**: alternatives to the standard
picture, candidate theories that no experiment has tested yet, and personal hypotheses. Every entry follows the same
rules, so the Lab is a place to test ideas in the open, not a place to promote them.

## The rules

1. **State the postulates precisely.** Write them as equations or definitions in the first problem of the series
   (role `postulate`). Vague ideas can't be tested.
2. **Reproduce what is known first.** Before predicting anything new, show that the idea gives back the established
   result in the regime where that result is measured (Newtonian gravity, the Born rule, ΛCDM distances, and so on).
3. **Make at least one numerical prediction that could fail.** A number with an uncertainty, not a story.
4. **Test it against open data** (role `experiment` or `observation`): bundle the data in `data/` with its source,
   licence and citation, fit it with tensors, and **show the result even when it goes against the idea.**
5. **Label it.** Every problem in the series carries the `speculative` tag (🧪 badge), and its summary says plainly
   what is speculative and what experiments say.
6. **Credit quietly.** The author goes in the series description in `content/series.json`, not in problem titles.

Leave parameters such as the number of dimensions open and fit them; don't hard-code the answer you hope for.

## How a series is wired

Each problem joins the series with tags in its `meta.yaml`:

```yaml
tags: [series:projection-hypothesis:1, role:postulate, speculative]
```

- `series:<id>:<step>`: the series id and the problem's position (1, 2, 3, …).
- `role:<kind>`: `postulate`, `theory`, `experiment`, `observation`, or for mathematics `definition`, `conjecture`,
  `theorem`, `proof`; `background` for context.
- `speculative`: the 🧪 badge.

The series itself is described once in `content/series.json`:

```json
"projection-hypothesis": {
  "group": "hypothesis-lab",
  "author": "…",
  "title": {"en": "…", "zh-Hant": "…", "zh-Hans": "…"},
  "summary": {"en": "…", "zh-Hant": "…", "zh-Hans": "…"}
}
```

Missing languages fall back to English. Run `pixi run python scripts/build_assets.py --manifest-only` and the series
appears under 🧪 Hypothesis Lab in the collections box, with a "step k of n" navigator on each of its problems.

## A typical story

| Step | Role | What it does |
|---|---|---|
| 1 | postulate | The idea as precise postulates, and an estimate of what it needs (scale, dimension, …) |
| 2–4 | theory | Consequences, each checked against a known limit |
| 5 | experiment / observation | Fits to open data, with the verdict: supported, constrained or ruled out |
