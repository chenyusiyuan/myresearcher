# Repository Guidelines

## Project Structure & Module Organization
This repository is split into a FastAPI/LangGraph backend and a Vue 3 frontend. Backend code lives in `backend/src/`, with graph orchestration under `backend/src/graph/`, integrations in `backend/src/services/`, and shared config/models in files such as `backend/src/config.py` and `backend/src/models.py`. Frontend code lives in `frontend/src/`; `App.vue` contains the main research UI and `services/api.ts` wraps HTTP calls. Operational scripts are in `scripts/`, logs are written to `logs/`, and local PID state is stored in `.run/`.

## Build, Test, and Development Commands
Install and run each app from its own directory:

- `cd backend && pip install -e .` installs the backend package.
- `cd backend && python src/main.py` starts the API on `127.0.0.1:8000`.
- `cd frontend && npm install` installs Vite/Vue dependencies.
- `cd frontend && npm run dev` starts the UI on `http://localhost:5174`.
- `cd frontend && npm run build` runs `vue-tsc` and produces a production build.
- `./scripts/start-all.sh`, `./scripts/status-all.sh`, and `./scripts/stop-all.sh` manage both services together.
- `cd backend && ruff check src` is the expected Python lint pass when `dev` dependencies are installed.

## Coding Style & Naming Conventions
Follow the existing style rather than reformatting unrelated code. Use 4-space indentation in Python and 2-space indentation in Vue/TypeScript/CSS. Keep Python modules and functions `snake_case`, Pydantic models and enums `PascalCase`, and Vue components `PascalCase`. Prefer explicit types in TypeScript; `frontend/tsconfig.json` enables `strict` mode. Backend linting is configured with Ruff and Google-style docstrings.

## Testing Guidelines
There are no committed automated tests yet. For backend changes, add `pytest`-style tests under `backend/tests/` with filenames like `test_config.py`. For frontend logic, add `*.test.ts` files alongside the code or under `frontend/src/`. Until a full test suite exists, contributors should at minimum run `npm run build`, start both services, and smoke-test `GET /healthz` plus a `/research` or `/research/stream` request.

## Commit & Pull Request Guidelines
Match the current history: short, imperative commit subjects such as `Implement multi-agent supervisor workflow` or `Refine reviewer missing topic normalization`. Keep commits focused on one change set. PRs should describe the user-visible impact, list any env/config changes, link related issues, and include screenshots or sample request/response snippets when the UI or streaming output changes.

## Security & Configuration Tips
Do not commit secrets. Keep local credentials in `backend/.env`; `backend/.env.example` is the template. Ignore generated files such as `frontend/dist/`, `frontend/node_modules/`, backend virtualenvs, and log output unless a change explicitly requires them.
