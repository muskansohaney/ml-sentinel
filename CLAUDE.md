# Project Guidance

ML Sentinel is a local-first MLOps reliability platform. Keep local development functional without AWS and favor small, focused changes that fit the existing project.

## Workflow

- Work through requested steps one at a time. Stop and report results before moving to the next step.
- Inspect relevant code and documentation before changing behavior; preserve existing patterns.
- Run `pytest -q` after code changes. Use the project virtual environment and clear ROS-provided `PYTHONPATH` and `AMENT_PREFIX_PATH` if they interfere.
- Verify changes with appropriate checks before reporting that they work.
- Add dependencies only when they are necessary.
- Make meaningful commits. Never push without asking first.

## Safety and Cost

- Do not run destructive commands.
- Do not perform real model promotion or rollback.
- Do not create AWS resources without explicit approval.
- Minimize AWS costs and keep cloud services optional for local development.
