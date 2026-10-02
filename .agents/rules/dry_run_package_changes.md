# Rule: Dry Run Package & Dependency Changes Before Committing

## Directive
- Before committing any package, dependency, or version changes (e.g. `pyproject.toml`, `requirements.txt`, or setup configs), perform a dry run locally to verify syntax, imports, and package compatibility without forcibly reinstalling all packages or breaking existing virtual environments.
- NEVER use `git reset --hard` under any circumstances.
