# Repository Guidelines

## Skills

The `skills/` directory contains structured guides for common tasks (running
tests, building containers, managing dependencies, submitting SLURM jobs, etc.).
**Always read the relevant `SKILL.md` before starting any task it covers —
skills are mandatory context, not optional background reading.**

**Workflow — mandatory order for every task:**
1. **Pull information first.** Read the commit, PR, error log, file, or
   whatever artifact the task is about. Do not reason about it yet.
2. **Select and invoke the skill.** Based on what you just read, identify
   the relevant skill and invoke it before forming any answer or plan.
3. **Answer or implement.** Only after the skill is loaded, use its context
   to reason, diagnose, or write code.

Never skip or reorder these steps. Do not wait for the user to name the right
skill keyword — infer it from the artifact you read.

## Contributing

### Pull Requests

- All PRs must be created as **drafts**. Use `gh pr create --draft` or the GitHub UI draft option.
- Never push branches directly to `https://github.com/NVIDIA/Megatron-LM`. You must push your branch to a personal fork (e.g. `https://github.com/<your-username>/Megatron-LM`), then open a PR from the fork's branch against `NVIDIA/Megatron-LM`.
- Commit PR changes with both `-s` and `-S`: `-s` adds the required
  `Signed-off-by` trailer, and `-S` signs the commit so copy-pr-bot and `/ok to
test` can verify the pushed commit without manually specifying the SHA.
Megatron Core engineers at NVIDIA should sign using their NVIDIA emails so they
are automatically added to the right user groups on the internal Slack
workspace.
- Read @docs/developer/contribute.md for the full contribution policy, including code style, commit message conventions, and issue guidelines.

### Code Quality

- After editing imports in any Python files, always run `uv run isort` on those files to fix import order before committing.

### Megatron Core Process Groups

- In `megatron/core` production code, avoid adding new direct reads of global
  process groups from `parallel_state` (for example,
  `parallel_state.get_tensor_model_parallel_group()` or directly imported
  `get_*_group()` helpers). Prefer accepting a `ProcessGroupCollection` or an
  explicit `torch.distributed.ProcessGroup` from the caller and passing that
  through.
- Allowed compatibility points include `megatron/core/parallel_state.py`,
  `megatron/core/process_groups_config.py`, initialization/bootstrap code that
  materializes a `ProcessGroupCollection` from MPU globals, tests, docs, and
  migration fallbacks with an explicit comment.
- This guidance targets Megatron Core library code. Do not apply it to
  `megatron/training` or other training-loop code unless the PR explicitly
  opts into that migration.
- In reviews, flag new direct `parallel_state.get_*_group()` usage in
 `megatron/core` unless it is one of the compatibility points above. This is
 advisory guidance, not a CI gate.

## Cursor Cloud specific instructions

The Cursor Cloud VM is **CPU-only (no GPU, no Docker/NGC container)**. Megatron
is designed to run inside the NGC PyTorch CI container (see
`skills/mcore-build-and-dependency/SKILL.md`), which supplies CUDA, NCCL, and the
prebuilt native extensions. That full GPU/container path is **not reproducible
here**; the cloud VM only supports the CPU-friendly subset described below.

### What the startup update script sets up
- Creates/refreshes the uv-managed venv at `.venv` via `uv sync --locked`
  (default groups: `linting`, `build`, `test`). This installs lint/test tooling
  and the editable `megatron-core` package.
- `pyproject.toml` `[tool.uv].override-dependencies` strips `torch`, `triton`,
  and `torchvision` on every platform (the CI container normally provides them),
  and `uv sync` **removes** any that aren't in the lock. The update script
  therefore reinstalls CPU wheels *after* sync with
  `uv pip install --no-config torch triton --torch-backend=cpu`. `--no-config`
  is required to bypass the override. Never reorder these: sync first, then the
  torch/triton reinstall.

### Non-obvious caveats
- Use `.venv/bin/python` directly. Avoid `uv run …`: it triggers an implicit
  `uv sync` that deletes the CPU torch/triton wheels and breaks imports.
- Importing `megatron.core` prints harmless "Transformer Engine and Apex are not
  installed" warnings — expected on CPU (it falls back to Torch implementations).
- The optional C++ extension `megatron/core/datasets/helpers_cpp` is not built by
  `uv sync`. To build it (needs the system `python3-dev` headers, already
  installed in the snapshot):
  `cd megatron/core/datasets && make CXX=g++ "CPPFLAGS=$(../../../.venv/bin/python -m pybind11 --includes)" "LIBEXT=$(python3-config --extension-suffix)"`.
- Lint runs on CPU: `.venv/bin/ruff check megatron/core` (the repo enables only
  the `S506` rule; expect "All checks passed"). Full formatter:
  `BASE_REF=main CHECK_ONLY=true bash tools/autoformat.sh`.
- The unit/functional test suites are **GPU + container bound** and not runnable
  here: `tests/unit_tests/test_utilities.py` hardcodes the NCCL backend and
  `torch.cuda.set_device`, `conftest.py` expects test data at `/opt/data`, and
  many tests import `transformer_engine` or download HF/nltk assets over the
  network. Do not expect `pytest tests/unit_tests` to pass on the cloud VM.
- A full transformer forward pass also requires a GPU: the local
  `DotProductAttention` global memory buffer calls `torch.cuda.current_device()`.
  CPU-safe smoke checks are limited to model construction (`use_cpu_initialization=True`),
  the input embedding forward, and tensor-parallel linear layer forwards.
