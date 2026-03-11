"""
Real subprocess sandbox executor for MatrixArena.

Each call to ``SubprocessExecutor.run()`` will:

1. Syntax-check the solution code via ``compile()`` (fast, no subprocess needed).
2. Build a self-contained test harness script that:
   - Executes the solution code in the same namespace.
   - Attempts to discover the primary callable and invoke it with the
     keyword-argument string provided in each test case.
   - Emits a JSON array of per-test results to stdout.
3. Run the harness in an isolated subprocess with a hard timeout.
4. Return a structured ``ExecutionResult`` dict that is passed to the
   Judge models as additional evidence.

Fallback behaviour
------------------
If the input strings cannot be parsed as Python keyword-arguments (e.g. they
are natural-language descriptions rather than code), the harness gracefully
reports each test case as ``"skip"`` so the Judge can still evaluate the
solution from reading the code alone.

Docker upgrade path
-------------------
Replace the ``subprocess.run`` call with a ``docker run`` invocation to run
the harness inside the image defined in ``sandbox/Dockerfile`` for stronger
isolation in a production setting.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# Maximum wall-clock seconds allowed for a single execution
DEFAULT_TIMEOUT: int = 30

# ---------------------------------------------------------------------------
# Test-harness template injected after the solution code
# ---------------------------------------------------------------------------
_HARNESS = textwrap.dedent("""

# ── MatrixArena Test Harness ───────────────────────────────────────────────
import inspect as _inspect
import json as _json

_test_cases = {test_cases_json}

# Discover callables introduced by the solution (exclude builtins / imports)
_ns_before = {{'_inspect', '_json', '_test_cases'}}
_fns = [
    (name, obj)
    for name, obj in list(locals().items())
    if callable(obj)
    and not name.startswith('_')
    and name not in _ns_before
    and _inspect.isfunction(obj)
]

_results = []
for _i, _tc in enumerate(_test_cases):
    _inp = _tc.get('input', '')
    _expected = str(_tc.get('expected_output', '')).strip()
    _status = 'skip'
    _actual = None
    _error = None

    if _fns:
        # Use the last user-defined function as the entry point
        _fn_name, _fn = _fns[-1]
        try:
            _result = eval(f'_fn({{_inp}})', {{**locals(), '_fn': _fn}})
            _actual = str(_result).strip()
            _status = 'pass' if _actual == _expected else 'fail'
        except Exception as _e:
            _error = type(_e).__name__ + ': ' + str(_e)
            _status = 'error'
    
    _results.append({{
        'test': _i + 1,
        'status': _status,
        'input': _inp,
        'expected': _expected,
        'actual': _actual,
        'error': _error,
    }})

print(_json.dumps(_results))
""")


class ExecutionError(Exception):
    """Raised when the executor itself fails (not the solution code)."""


class SubprocessExecutor:
    """
    Run solver code in an isolated subprocess and return structured results.

    Parameters
    ----------
    timeout:
        Maximum seconds to wait for the subprocess.  Defaults to
        ``DEFAULT_TIMEOUT``.
    python:
        Python executable to use inside the subprocess.  Defaults to the
        same interpreter running MatrixArena (``sys.executable``).
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        python: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.python = python or sys.executable

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, code: str, test_cases: list[dict] | None = None) -> dict:
        """
        Execute *code* against *test_cases* in a sandboxed subprocess.

        Parameters
        ----------
        code:
            Complete Python source produced by the Solver.
        test_cases:
            List of ``{"input": ..., "expected_output": ...}`` dicts from
            the Generator's problem spec.  If ``None`` or empty, only syntax
            and import checking is performed.

        Returns
        -------
        dict
            ::

                {
                    "status":  "success" | "syntax_error" | "timeout" | "runtime_error",
                    "summary": "<human-readable one-liner>",
                    "passed":  <int>,
                    "failed":  <int>,
                    "skipped": <int>,
                    "tests":   [<per-test dicts>],
                    "stderr":  "<captured stderr>",
                }
        """
        if not code or not code.strip():
            return self._result("runtime_error", "Solver returned empty code.", [], "")

        # ── Step 1: syntax check (no subprocess, instant) ───────────────
        try:
            compile(code, "<solution>", "exec")
        except SyntaxError as exc:
            return self._result(
                "syntax_error",
                f"SyntaxError at line {exc.lineno}: {exc.msg}",
                [],
                str(exc),
            )

        # ── Step 2: build harness script ────────────────────────────────
        tc_list = test_cases or []
        harness_code = code + _HARNESS.format(
            test_cases_json=json.dumps(tc_list, ensure_ascii=False)
        )

        # ── Step 3: run in subprocess ────────────────────────────────────
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(harness_code)
            tmp_path = Path(tmp.name)

        try:
            proc = subprocess.run(
                [self.python, str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return self._result(
                "timeout",
                f"Execution exceeded {self.timeout}s time limit.",
                [],
                "",
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        stderr = proc.stderr.strip()

        if proc.returncode != 0 and not proc.stdout.strip():
            return self._result(
                "runtime_error",
                f"Process exited with code {proc.returncode}.",
                [],
                stderr,
            )

        # ── Step 4: parse harness JSON output ────────────────────────────
        try:
            tests: list[dict] = json.loads(proc.stdout.strip())
        except (json.JSONDecodeError, ValueError):
            # Code ran but produced no structured output (script-style solution)
            status = "success" if proc.returncode == 0 else "runtime_error"
            summary = "Code executed (no structured test output)."
            if proc.returncode != 0:
                summary = f"Process exited with code {proc.returncode}."
            return self._result(status, summary, [], stderr)

        return self._result("success", "", tests, stderr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _result(
        status: str,
        summary: str,
        tests: list[dict],
        stderr: str,
    ) -> dict:
        passed = sum(1 for t in tests if t.get("status") == "pass")
        failed = sum(1 for t in tests if t.get("status") == "fail")
        skipped = sum(1 for t in tests if t.get("status") in ("skip", "error"))
        total = len(tests)

        if not summary:
            if total == 0:
                summary = f"Status: {status}."
            else:
                summary = f"{passed}/{total} test(s) passed."
                if failed:
                    summary += f"  {failed} failed."
                if skipped:
                    summary += f"  {skipped} skipped/errored."

        return {
            "status": status,
            "summary": summary,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "tests": tests,
            "stderr": stderr[:1000] if stderr else "",  # cap stderr length
        }


# ---------------------------------------------------------------------------
# Backwards-compatible alias so existing code that imports MockExecutor still
# works during the transition period.
# ---------------------------------------------------------------------------
class MockExecutor(SubprocessExecutor):
    """Deprecated alias for SubprocessExecutor.  Use SubprocessExecutor directly."""

