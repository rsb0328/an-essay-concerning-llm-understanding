# Continuous integration template

The publishing token used for the initial repository upload did not have GitHub's separate `workflow` scope, so the initial release does not activate a workflow automatically.

After granting that scope, save the following as `.github/workflows/test.yml`:

```yaml
name: tests

on:
  push:
  pull_request:

jobs:
  unit:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - run: pip install -e .
      - run: python -m unittest discover -s tests -v
```

Before enabling it, authenticate GitHub CLI with workflow permission:

```bash
gh auth refresh -h github.com -s workflow
```
