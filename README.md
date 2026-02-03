### External dependency

This project uses the upstream repository `hyperbolic-fully-connected` as a Git submodule so we can:
- reuse their layers directly
- easily pull upstream fixes/improvements

The package is installed in editable mode, so imports work normally while still pointing to the submodule source.

#### Initial setup

Clone with submodules:
```bash
git clone --recurse-submodules <repo-url>
```
or, if already cloned:
```bash
git submodule update --init --recursive
```

#### Install the package

From the repo root:
```bash
pip install -e external/hyperbolic-fully-connected --no-deps
```
Notes:
-e = editable install (imports directly from the submodule folder)
--no-deps = dependencies are managed by our environment .yml

#### Updating to latest upstream
Pull newest commits from the original repository:

git submodule update --remote --merge
git commit -m "Update hyperbolic-fully-connected submodule"