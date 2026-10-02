# Rule: Never Use Git Hard Reset

## Directive
- **NEVER** use `git reset --hard` or perform destructive hard resets on any git branches or worktrees under any circumstances.
- Always preserve user and working directory state using standard merges, rebase, clean stashes, or fast-forward commits (`git pull`, `git merge`, `git checkout -b`).
