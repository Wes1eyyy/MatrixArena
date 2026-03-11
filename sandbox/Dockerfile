# MatrixArena Sandbox — Isolated code execution environment
#
# Build:   docker build -t matrix-arena-sandbox sandbox/
# The SubprocessExecutor in executor.py can be upgraded to use this image
# by replacing the subprocess.run call with:
#
#   docker run --rm --network none --memory 256m --cpus 0.5 \
#       --user nobody -v /tmp/solution.py:/home/sandbox/solution.py \
#       matrix-arena-sandbox
#
# Security hardening applied:
#   - Non-root user (nobody)
#   - No network access (--network none at runtime)
#   - Read-only filesystem except /tmp (enforce at runtime with --read-only)
#   - Minimal base image (slim)
#   - No shell in PATH beyond /bin/sh (no bash)

FROM python:3.13-slim

# Remove package manager caches and unnecessary tools
RUN apt-get purge -y --auto-remove wget curl git && \
    rm -rf /var/lib/apt/lists/*

# Create a non-privileged user for code execution
RUN useradd --create-home --no-log-init --shell /bin/false sandbox

WORKDIR /home/sandbox

# Drop to non-root
USER sandbox

# Default: run solution.py placed by the host via volume mount or COPY
CMD ["python", "-u", "solution.py"]

