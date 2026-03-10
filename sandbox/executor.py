"""
Mock executor for the MatrixArena MVP sandbox.

In a production system this would spin up an isolated Docker container,
copy the solver's code into it, run the test cases, and return pass/fail
results.  For the MVP we simply return a placeholder result so the rest
of the pipeline can function without a real execution environment.
"""

from __future__ import annotations


class MockExecutor:
    """
    Stub code executor that always reports 'Pass'.

    Replace the ``run`` method body with real sandbox logic (e.g. a Docker
    container call) to enable actual code execution.
    """

    def run(self, code: str) -> str:  # noqa: ARG002
        """
        Simulate executing *code* and return a status string.

        Parameters
        ----------
        code:
            The Python source code to execute (ignored in MVP).

        Returns
        -------
        str
            Always ``"Pass"`` in the MVP stub.
        """
        return "Pass"
