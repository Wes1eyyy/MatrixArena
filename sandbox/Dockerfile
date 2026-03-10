# MatrixArena Sandbox — Dockerfile (Placeholder)
#
# This Dockerfile defines the isolated execution environment that will be
# used in production to safely run solver-generated code.
#
# For the MVP, the MockExecutor in executor.py is used instead.
#
# Future implementation sketch:
#   1. Base image: python:3.12-slim
#   2. Create a non-root user for code execution
#   3. Set strict resource limits (ulimit) and network isolation
#   4. COPY the solver code in at runtime via a volume mount or stdin
#   5. Run the code with a timeout (e.g. `timeout 10 python solution.py`)

FROM python:3.12-slim

# Create a non-privileged user
RUN useradd --create-home sandbox
USER sandbox
WORKDIR /home/sandbox

# Default command — overridden at runtime
CMD ["python", "solution.py"]
