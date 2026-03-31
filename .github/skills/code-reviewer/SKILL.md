---
name: code-reviewer
description: Review repository code changes with a findings-first focus on correctness, regressions, compatibility, and missing validation. Use for code review, risk review, or model adaptation review requests.
license: Apache-2.0
metadata:
  author: Lihui
  version: "3.0"
---

# Code Reviewer

Use this skill when the user asks for code review, risk review, regression review, or quality review.

## Scope

- Review only. Do not edit code unless the user explicitly switches from review to implementation.
- Prioritize real defects and behavior risks over style-only comments.
- Keep the review grounded in the requested scope. If the user names files, models, or directories, do not expand the scope without saying so.

## Review Priorities

Review in this order:

1. Correctness and behavioral regressions
2. Compatibility with existing APIs, configs, and model registration
3. Cache, shape, dtype, and sequence-length contracts
4. Reliability, error handling, and edge cases
5. Missing or weak validation
6. Maintainability and readability

## Repository-Specific Checks

- For code under `xhmodel_merak/xh_llm/models`, check registration, wrapper replacement, cache flow, and output semantics.
- For CLI changes, check command routing, argument compatibility, and help behavior.
- For config or example changes, check that names, paths, and parameters still match the code they exercise.
- If the review is about model adaptation compatibility, also read `.github/skills/xhquant_llm_reviewer/SKILL.md`.

## Model Review Routing

When the user asks to review a specific adapted model, default review targets are:

- `xhmodel_merak/xh_llm/models/<model_name>/`
- `examples_merak/llm/<model_name>/` if it exists

When the user asks for a broad LLM review, enumerate model directories under `xhmodel_merak/xh_llm/models/` and review them one by one instead of giving a vague aggregate answer.

## Output

Return findings first, ordered by severity.

For each finding include:

- file or symbol
- concrete issue
- impact
- minimal fix direction

Then include:

- open questions or assumptions
- residual risk or validation gaps
- a short overall conclusion

If there are no findings, say that explicitly and still mention residual risks or unverified areas.
