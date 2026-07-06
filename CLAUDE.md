# CLAUDE.md

Full repo guide: [.github/copilot-instructions.md](.github/copilot-instructions.md)
(fork identity, non-negotiable rules, layout, merge workflow, conflict rules).
Route by what you are about to do:

- Touching anything -> read §1 "Fork Identity & Non-Negotiable Rules" first.
- About to merge upstream -> §3.1 preparation, then §3.2 conflict triage and
  §3.3 resolution order. NEVER improvise conflict resolution -> §4 names the
  rule for each file category (§7 is the file-by-file quick reference).
- Merge conflicts all resolved -> run every §3.4 post-merge validation step
  (source-diff helper, build, orphaned conflict-marker grep, xla_clang.patch
  intactness checks: `.cu` extensions in device files, const removal in
  generate.py) and §8 install-tree comparison; report each pass/fail.

## Working with the user

NEVER delete or rewrite the user's "Prompt" lines (lines they wrote as
prompts/questions) -> reproduce them verbatim and put your response alongside.
