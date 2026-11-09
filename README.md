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
