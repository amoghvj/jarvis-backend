# J.A.R.V.I.S. Backend

FastAPI backend: GitHub login, account API keys, and authenticated telemetry ingestion (`POST /v1/events`). Events are still only printed to the console.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate      # PowerShell
pip install -r requirements-dev.txt
cp .env.example .env                                # fill in real values locally (git-ignored)
```

Run the database migration once (Supabase SQL editor): [migrations/001_create_sessions.sql](migrations/001_create_sessions.sql). It is additive (`create table if not exists public.sessions`) and is required because the existing schema has nowhere to store revocable sessions.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload   # docs at /docs
pytest                                                      # see "Tests"
```

## Environment variables

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Supabase **Transaction Pooler** connection string |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | GitHub OAuth App credentials |
| `GITHUB_CALLBACK_URL` | `https://<backend-domain>/auth/github/callback` |
| `FRONTEND_URL` | Future frontend origin (CORS, CSRF origin check, optional redirect) |
| `SESSION_SECRET` | Random string, 32+ chars (signs OAuth state). Generate: `python -c "import secrets;print(secrets.token_urlsafe(48))"` |
| `POST_LOGIN_REDIRECT` | Optional, default `false`: callback returns a JSON success response. Set `true` once the frontend exists to redirect to `FRONTEND_URL` |
| `COOKIE_SAMESITE` | Optional override: `lax`, `strict`, `none` |

## GitHub OAuth App

- Authorization callback URL: exactly the value of `GITHUB_CALLBACK_URL`.
- Scopes requested: `read:user user:email` only.

## Cookies (environment-aware)

| Environment | Cookie |
|---|---|
| Production (`GITHUB_CALLBACK_URL` is `https`), frontend on a different origin | `HttpOnly; Secure; SameSite=None` |
| Production, same origin | `HttpOnly; Secure; SameSite=Lax` |
| Local HTTP backend | `HttpOnly; SameSite=Lax` (no `Secure`) |

The cookie belongs to the backend's host. A frontend on `localhost` calling the deployed backend is cross-site, and browsers may block third-party cookies, so do not expect it to work. For local frontend development either run the backend locally too, or use a dev-server proxy so the browser only talks to one origin. Production is most reliable with frontend and API on one registrable domain (e.g. `app.example.com` and `api.example.com`, plus `COOKIE_SAMESITE=lax`).

CSRF: `POST /auth/logout`, `POST /api-keys/rotate` and `DELETE /api-keys` require an `Origin` header equal to `FRONTEND_URL` (or the backend's own origin). Browsers set this automatically. For curl, pass `-H "Origin: <FRONTEND_URL>"`.

## API

### `GET /auth/github`
Redirects (302) to GitHub with a random `state`; also sets a short-lived signed `jarvis_oauth_state` cookie. Open it in a browser.

### `GET /auth/github/callback?code=...&state=...`
Validates state, exchanges the code, upserts the user by GitHub ID, creates a session, sets the `jarvis_session` cookie. Temporary success response:
```json
{"status":"success","message":"Logged in with GitHub","new_account":true,
 "user":{"id":"…","github_id":123,"github_username":"octocat","email":"octo@example.com","avatar_url":"https://…","created_at":"…"}}
```
Errors: `400` invalid/expired state, GitHub denial, or missing code; `502` GitHub unreachable or rejected the code.

### `GET /auth/me`
`200` returns the user object above; `401 {"detail":"Not authenticated"}`.

### `POST /auth/logout`
Deletes the server-side session and clears the cookie. `200 {"status":"success","message":"Logged out"}`; `403` on bad Origin.

### `POST /api-keys/rotate`
```json
{"api_key":"jarvis_live_…","key_prefix":"jarvis_live_AbCd","created_at":"…","message":"Store this key now. It will not be shown again."}
```
The raw key is returned only here. `401` unauthenticated, `403` bad Origin.

### `GET /api-keys/status`
```json
{"active":true,"key_prefix":"jarvis_live_AbCd","created_at":"…","last_used_at":null}
```
`{"active":false,…null}` when no key exists.

### `DELETE /api-keys`
`200 {"status":"success","message":"API key revoked"}` (or `"No active API key"`).

### `POST /v1/events`
Same payload and `200` response as before, now with:
```
Authorization: Bearer jarvis_live_…
```
| Status | Meaning |
|---|---|
| 401 | Missing, invalid, revoked or rotated key |
| 403 | `application.project_id` belongs to another account |
| 404 | `application.project_id` does not exist (projects are **not** auto-created) |
| 422 | Invalid payload |

A batch is accepted or rejected as a whole. `last_used_at` is refreshed at most once a minute.

## Vercel

1. Project → Settings → Environment Variables: set every variable above for Production. Use the Transaction Pooler `DATABASE_URL` (port 6543).
2. `GITHUB_CALLBACK_URL` must match the OAuth App's callback URL exactly.
3. Push to the connected branch (or `vercel --prod`) to redeploy; env changes only apply to new deployments.
4. Verify: `GET /health`, then open `https://<backend>/auth/github` in a browser.

## Tests

`pytest` runs against the configured `DATABASE_URL` but inside a single transaction that is always rolled back (including the `sessions` migration), so no data persists. GitHub is mocked. Tests are skipped if `DATABASE_URL` is unset.


## GitHub App integration (installation linking, phases 1-3)

Lets a logged-in user install the J.A.R.V.I.S. **GitHub App** and link the installation to their account. This is separate from the GitHub **OAuth App** used for login and uses its own `GITHUB_APP_*` credentials. No repository access (tree, files, search, commits) is implemented yet.

### Flow
1. Frontend sends the user to `GET /integrations/github/install` (needs the J.A.R.V.I.S. session cookie). A pending record is stored with only a SHA-256 hash of a random token, and the browser is redirected to `https://github.com/apps/<slug>/installations/new`.
2. After installation GitHub redirects to `GET /integrations/github/setup?installation_id=…&setup_action=…`. The backend requires the session, validates the ID, stores a pending record (hashed random state, 10-minute expiry, single use) and redirects to GitHub's OAuth authorize URL. The state is also bound to the browser by a signed HttpOnly cookie.
3. GitHub redirects to `GET /integrations/github/callback`. The backend consumes the state, exchanges the code with the App's client secret, fetches `/user`, and checks the GitHub numeric ID equals `users.github_id`. It then checks that the installation is in `/user/installations`, belongs to this App and is not suspended. It stores the link in a transaction and discards the user token.
4. The browser is redirected to `FRONTEND_URL?github_app=connected&installation_id=…`, or `?github_app=error&reason=<code>`. Codes, tokens and keys never appear in the URL.

`reason` values: `invalid_state`, `expired_state`, `replayed_state`, `cancelled`, `missing_params`, `identity_mismatch`, `installation_not_accessible`, `installation_suspended`, `installation_already_linked`, `github_oauth_error`.

### Other endpoints
- `GET /integrations/github/status` returns `{"connected": bool, "installations": [{installation_id, account_id, account_login, account_type, repository_selection, permissions, suspended_at, created_at, updated_at}]}` for the current user only.
- `DELETE /integrations/github/installations/{installation_id}` (needs the `Origin` header, same CSRF rule as other mutations) removes **only the local link**. It does **not** uninstall the App from GitHub; users must do that in GitHub settings (Settings → Applications → Installed GitHub Apps). Another user's installation returns 404.

An installation can be linked to only one J.A.R.V.I.S. user; a conflicting link is rejected.

### Migration
Run `migrations/002_create_github_app_installations.sql` in the Supabase SQL editor (additive, idempotent, RLS enabled, no policies). It creates `github_app_installations` and `github_app_pending_installations`.

### GitHub App settings (github.com → Settings → Developer settings → GitHub Apps)
- Setup URL: `https://jarvis-backend-kappa-six.vercel.app/integrations/github/setup`
- Callback URL: `https://jarvis-backend-kappa-six.vercel.app/integrations/github/callback`
- **Do not** enable "Request user authorization (OAuth) during installation"
- Repository permissions: Metadata (read-only), Contents (read-only)
- Webhooks: disabled
- Generate a client secret and a private key (`.pem`)

### Environment variables (local `.env` and Vercel)
| Variable | Value |
|---|---|
| `GITHUB_APP_ID` | Numeric **App ID** (not the Client ID) |
| `GITHUB_APP_CLIENT_ID` | App's Client ID (`Iv1.…`) |
| `GITHUB_APP_CLIENT_SECRET` | App's client secret |
| `GITHUB_APP_PRIVATE_KEY` | Full PEM. Multiline is fine; otherwise one line with `\n` for newlines |
| `GITHUB_APP_SLUG` | The App's URL slug |
| `GITHUB_APP_CALLBACK_URL` | The callback URL above |

Never commit the key or secret, and never put them in frontend variables. Redeploy after changing Vercel variables.

### Status of this feature
- **Implemented and unit-tested** (mocked GitHub, real SQL in a rolled-back transaction): App JWT, installation-token creation, flow state, replay and expiry rejection, identity mismatch, ownership checks, duplicate and cross-user handling, status, disconnect.
- **Not tested**: any real GitHub installation or OAuth round trip, the real GitHub API response shapes, behaviour on Vercel, and migration 002 against your database.


## Investigation tools: shared infrastructure and GitHub repository tools

Reusable, read-only tools for the future investigation agent live in `app/tools/`. There is **no HTTP route, agent loop, LLM call or chat integration** yet; callers invoke tools in-process:

```python
from app.tools.executor import execute_tool
from app.tools.schemas import ToolContext

ctx = ToolContext(user_id=user["id"], conn=conn)   # user comes from a verified session, never from input
result = execute_tool("get_file_contents",
                      {"installation_id": 123, "owner": "acme", "repo": "webapp", "path": "src/app.py"}, ctx)
```

`app.tools.definitions.get_default_registry().describe()` returns machine-readable definitions (name, description, JSON input/output schemas, constraints).

Every call returns the same shape: `{tool_name, status: "success"|"error", data, metadata: {duration_ms, source, github_requests}, error: {code, message, retryable} | null}`. Unknown fields in tool input are rejected (so `user_id` cannot be supplied), raw exceptions are logged server-side with credentials redacted, and results never contain tokens, stack traces or SQL.

| Tool | Purpose | Notable bounds |
|---|---|---|
| `list_installations` | The user's linked installations (database only) | own installations only |
| `list_repositories` | Repositories in an installation | `page`, `per_page` ≤ 100 |
| `get_repository_metadata` | Identity, visibility, language, URLs, timestamps | |
| `get_repository_tree` | Directory listing at a reference | `limit` ≤ 1000, `offset`, reports GitHub truncation |
| `get_file_contents` | One UTF-8 text file | refuses files > 512 KB, binary and non-UTF-8 (never truncates) |
| `search_repository` | Literal text search at a reference | ≤ 100 files, 2 MB, 50 results; reports `limits_hit` and `complete` |
| `get_commit` | Author, date, message, parents, stats | no email addresses returned |
| `get_commit_diff` | Changed files with patches | 20k chars per patch, 120k total, paginated |
| `get_file_history` | Commits that touched a file | paginated |

**Authorization.** Each repository tool first proves the installation is linked to the authenticated user (a foreign and a nonexistent installation look identical), rejects suspended installations, then confirms the repository is in the installation's repository list *before* using it. That last check matters because an installation token can read any public repository. The GitHub App's token service is reused; the OAuth login token is never used. Tokens never leave `app/tools/github/api.py`.

**Versions.** `ref` accepts a branch, tag or commit SHA (7–40 hex characters are treated as a SHA). Without one, the default branch head is used and every result reports `reference` (resolved commit SHA, `default_branch_used`, and a warning that it may not match the code running during an incident). Content is always read at the resolved commit SHA.

**Limitations.** `search_repository` deliberately does not use GitHub code search (it only indexes the default branch and cannot be pinned to a commit). Installations with more than 1,000 repositories fail closed. Tool time limits (20–28 s) may exceed your Vercel function duration, so raise `maxDuration` before exposing tools over HTTP. None of this has run against real GitHub yet.

## Projects, telemetry persistence, incidents and telemetry tools

### Applying migration 003 (manual; never run automatically)

`migrations/003_project_telemetry_incidents.sql` is additive and idempotent. Statements that touch the
existing `projects` / `incidents` tables are guarded, so a repeat run takes no locks and changes nothing.

**Prerequisites:** `public.users`, `public.projects`, `public.incidents` (base schema), `public.sessions`
(migration 001), and `public.github_app_installations` with `UNIQUE (installation_id)` (migration 002) must
already exist. Run as the table owner (`postgres` in the Supabase SQL Editor).

1. Open the Supabase dashboard, then **SQL Editor**, then **New query**.
2. Paste the full contents of `migrations/003_project_telemetry_incidents.sql` and **Run** it (one transaction).
3. Verify:
   ```sql
   select conname, pg_get_constraintdef(oid) from pg_constraint
   where conrelid = 'public.projects'::regclass;          -- expect UNIQUE (user_id, project_id); no projects_project_id_key
   select conname, pg_get_constraintdef(oid) from pg_constraint
   where conrelid in ('public.incidents'::regclass, 'public.telemetry_events'::regclass)
     and conname in ('incidents_status_check', 'telemetry_events_classification_check');  -- 2 rows
   select table_name from information_schema.tables
   where table_schema = 'public'
     and table_name in ('telemetry_events', 'project_repositories', 'incident_history');   -- 3 rows
   select relname, relrowsecurity from pg_class
   where relname in ('telemetry_events', 'project_repositories', 'incident_history');      -- all true
   ```
4. Deploy the backend **after** the migration (the new code needs the new tables and columns).

Rollback: there is no destructive step to undo. To go back to global uniqueness (only safe while no two users share
a `project_id`): `alter table public.projects add constraint projects_project_id_key unique (project_id);`
then drop `projects_user_id_project_id_key`.

### Project API (session cookie; state changes also need a trusted `Origin`)

| Endpoint | Purpose |
|---|---|
| `GET /projects` | The caller's projects. |
| `POST /projects` `{project_id, name?}` | Create. `project_id`: 3-64 chars, letters/digits/`.`/`_`/`-`, unique **per user** (409 on duplicate). The owner is always the session user; extra fields are rejected. No repository mapping is created. |
| `GET /projects/{id}` | Details (`id` = internal UUID). Someone else's project is a 404. |
| `PATCH /projects/{id}` `{name}` | Rename. |
| `GET /projects/{id}/incidents?status=&limit=&offset=` | Incidents, newest activity first. |
| `POST /projects/{id}/incidents/{incident_id}/status` `{status, note?}` | `open` -> `acknowledged`/`resolved`, `acknowledged` -> `resolved`/`open`, `resolved` -> `open`. Recorded in `incident_history`. Never automatic. |
| `GET /projects/{id}/repositories` | Repositories discovered from telemetry with their verification status. |
| `POST /projects/{id}/repositories/verify` | Re-check unverified repositories (for example after installing the GitHub App). |

### Telemetry ingestion (`POST /v1/events`, API-key authenticated)

* Events are persisted in `telemetry_events`, idempotent on `UNIQUE (project_id, event_id)`. A retried event
  creates no second row, no second incident count and no second startup registration.
* The response is unchanged (`status`, `message`, `received`) when every event was newly stored. If any event was a
  duplicate or rejected, the response also carries `accepted`, `duplicates`, `rejected` and `results`
  (per-event outcomes for the non-accepted ones); `status` is `partial` when something was rejected.
* Each event is validated independently, so one bad event does not discard the rest. A batch with no valid event is a
  422 listing why. Limits: 100 events per batch, 1,000,000 bytes per request, 100,000 bytes per event (413 / rejected).
* The project is resolved from `application.project_id` **and the API key's owner**. Unknown project is 404, a project
  that belongs to someone else is 403, and the whole batch is refused.
* The whole batch (events plus incident grouping) is one transaction: a failure stores nothing and the SDK can retry.

### Incident grouping

Eligible events are non-`startup`, non-`normal_behaviour` events with severity `ERROR` or `CRITICAL`. The fingerprint is
SHA-256 over project, service, environment, error type, lightly normalized message (whitespace and `0x...` addresses
only), source file and source function. Timestamps, request/trace/span/instance IDs and line numbers are excluded.
An existing `open`/`acknowledged` incident with the same fingerprint whose first/last-seen range is within the window
(`INCIDENT_CORRELATION_WINDOW_SECONDS`, default 900) absorbs the event; otherwise a new incident is created. A
transaction-scoped advisory lock per (project, fingerprint) makes this safe under concurrency. Severity only escalates.

### Repository discovery and verification

A `startup` event's `application.repository` registers an **unverified** row in `project_repositories`. It becomes
`verified` only if the existing installation-authorization path finds that exact repository in one of the project
owner's non-suspended GitHub App installations; `installation_id` is only ever set by that check. Verified rows are
re-checked against the installation on every read, so suspension or uninstalling stops them being offered immediately.

### Telemetry tools

`get_incident`, `get_related_events`, `search_events`, `get_event_details`, `get_trace`, `get_service_context` and
`get_incident_timeline` take an internal `project_id` that the shared executor injects from the trusted
`ToolContext.project_id` after verifying the user owns it. It is omitted from the LLM-facing schema
(`registry.describe()`; `describe(include_internal=True)` shows it) and rejected if a caller supplies it.

### Running the tests

The tests never touch the shared Supabase database by default. They start a throwaway local PostgreSQL (needs
`initdb`/`pg_ctl` on `PATH`, `PG_BIN`, or a standard Windows install), or use `TEST_DATABASE_URL`. Set
`JARVIS_TEST_USE_REMOTE_DB=1` to fall back to the legacy rolled-back-transaction mode against `DATABASE_URL`.

### Event classification (jarvis-observe 0.1.1 contract)

The canonical values are `normal_behaviour` and `abnormal_behaviour` (`telemetry_service.resolve_classification`).

| Incoming `classification` | Stored in `telemetry_events.classification` |
|---|---|
| field absent, `type` is not `startup` (SDK 0.1.0) | `abnormal_behaviour` |
| field absent, `type: startup` | `normal_behaviour` (a lifecycle event) |
| `normal_behaviour` / `abnormal_behaviour`, any casing, surrounding whitespace ignored | the lower-case canonical value |
| anything else: unknown or empty string, `null`, number, boolean, list, object | **event rejected** (per-event `rejected` result, or 422 if no event in the batch is valid); never stored, truncated or replaced |

* The classification column is always one of the two values, so the `NOT NULL` constraint cannot fail.
* The raw event is kept untouched in `payload`; only the column is normalized (for example `"NORMAL_BEHAVIOUR"` in the payload is
  `normal_behaviour` in the column).
* Valid but contradictory combinations are stored exactly as sent and are not reinterpreted: `startup` + `abnormal_behaviour` is stored
  as `abnormal_behaviour`, and `ERROR` + `normal_behaviour` is stored as `normal_behaviour`. Eligibility is decided separately:
  an event creates or joins an incident only if it is **not** `startup`, is classified `abnormal_behaviour` **and** has severity `ERROR` or
  `CRITICAL` (an `error`-type event with no severity at all is still treated as ERROR). INFO, DEBUG and WARNING never create incidents,
  even when classified `abnormal_behaviour`. The incident fingerprint does not include the classification.
