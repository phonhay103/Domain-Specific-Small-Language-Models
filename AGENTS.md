# AGENTS.md — Functional Programming Guidelines

Guidelines for AI agents and developers working in this repository.

## Core Paradigm

All scripts follow **Functional Programming (FP)** principles, keeping business logic pure and strictly isolated from side effects (I/O, ML execution, UI rendering).

## Functional Programming Rules

1. **Immutability First**
   - Model domain state using `@dataclass(frozen=True)`.
   - Prefer immutable collections (`tuple[T, ...]`) over mutable lists or dicts.
   - Never mutate objects in place; return new immutable data structures.

2. **Pure Functions & Composition**
   - Keep data transformations pure and deterministic.
   - Leverage `common.functional` primitives (`pipe`, `compose`, `map_tuple`, `filter_tuple`, `chunk_list`, `calculate_speedup`).
   - Require explicit type annotations on all function signatures (`Sequence`, `Callable`, `tuple`, etc.).

3. **Isolated Side Effects**
   - Separate pure computation from side effects (I/O, PyTorch/FAISS model inference, terminal outputs).
   - Isolate Rich UI terminal formatting into dedicated `render_*` functions.

4. **Verification Commands**
   - `make check` — Run Ruff linter, formatter checks, and `ty` type checker.
   - `make test-syntax` — Verify Python syntax compilation across all chapter scripts.
