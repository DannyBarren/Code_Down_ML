# Hosted client instance on Railway

This is the **client-ready** deploy of Code_Down_ML: a long-running Streamlit
process, a persistent volume at `/data`, and a shared password on the URL.

It is **not** the public Hugging Face demo. The existing `Dockerfile` sets
`FILLDOWN_DEMO=1` and wipes the knowledge base on start. Point Railway at
`Dockerfile.railway` on branch `demo/railway-live`.

Do not add Postgres. SQLite on the volume is the knowledge base.

## 1. New project from GitHub

1. In Railway, create a new project.
2. Deploy from GitHub repo `DannyBarren/Code_Down_ML`.
3. Select branch **`demo/railway-live`**.

## 2. Dockerfile

`railway.toml` already sets:

```
builder = DOCKERFILE
dockerfilePath = Dockerfile.railway
```

If Railway ignores the toml, open **Settings** and set the Dockerfile path to
`Dockerfile.railway` by hand.

Do **not** use the root `Dockerfile` — that image is the public demo
(`FILLDOWN_DEMO=1`, auto-wipe, XSRF off, port baked to 8501).

## 3. Persistent volume

Add a volume and mount it at **`/data`**.

That directory holds:

| Path | What |
| --- | --- |
| `/data/fill_down.db` | Rules, learned mappings, blocked pairings, training, account profiles, run history |
| `/data/models/` | Global + per-client LogReg weights (`clients/<safe_id>/`) |
| `/data/work/` | Temp work files |
| `/data/sessions/<safe_id>/` | Last coded workbook (`work_df.pkl` + original upload bytes) |

`safe_id` is `src.ml_classifier.safe_client_id` — e.g.
`Northwind Trading Co.` → `northwind_trading_co`.

## 4. Run as root so `/data` is writable

Railway mounts volumes as root. Set the service variable:

```
RAILWAY_RUN_UID=0
```

`Dockerfile.railway` does **not** switch to uid 1000 (the HF image does). A
uid-1000 process on a root-owned volume fails writes silently unless this is
set. If `/data` is not writable at boot, the app **refuses to start** and
shows the error in the UI — it will not fall through to `/tmp`.

## 5. Turn serverless / sleep OFF

In the service settings, turn **serverless / sleep / app hibernation OFF**.

Streamlit keeps a websocket open. When Railway freezes the process, the
browser disconnects and the conversion accountant loses the grid.

## 6. Variables

Set these on the service (no secrets in git):

```
FILLDOWN_LIVE=1
FILLDOWN_DEMO=0
FILLDOWN_DEMO_RESET=0
FILLDOWN_DATA_DIR=/data
FILLDOWN_MAX_ROWS=50000
FILLDOWN_AUTH_PASSWORD=<generate a long random password>
```

`FILLDOWN_AUTH_PASSWORD` is required. An empty password refuses to render the
app — there is no public unauthenticated client instance.

If someone also sets `FILLDOWN_DEMO=1`, live still wins: no wipe, no sample
auto-load, auth still applies.

Do not set `PORT` or `STREAMLIT_SERVER_PORT` yourself. Railway injects
`$PORT`; the image CMD is ` --server.port=${PORT:-8501}`.

## 7. Public URL

Generate a public URL. **Do not share it without the password.**

Hand the URL + password to the conversion accountant (and later, the client)
out of band.

## 8. Smoke (acceptance test)

After the service is up:

1. Open the URL. Log in with `FILLDOWN_AUTH_PASSWORD`.
2. Set **Client / project name** to e.g. `Northwind Trading Co.`
3. Upload `data/sample_transactions.csv` (or a real AppFolio / QuickBooks export).
4. Confirm the source-profile chip: `Read as AppFolio…` / `Read as QuickBooks…`
   / `Generic export`.
5. Click **Run Rules — Strict ★ (recommended)**.
6. Open **Review** and decide one leftover. Strict makes no suggestions, so
   the queue is blank rows: type an account and click **Fix**. (After a run
   that does suggest — Rules + Memory, Similarity, Full — the same card
   offers **Keep** and **Not this**.)
7. Optionally **Add another export** (a second small file).
8. Open **Insights** — cards render from the coded book.
9. **Restart** the Railway service.
10. Log in again. Type the same client name → **Resume last workspace**.
11. Confirm:
    - the workbook is back (filename + row count flash)
    - the code you kept / fixed is still there
    - rules / learned memory survived
    - Insights still render
    - **Add another export** still works
    - Excel download still has the original formatting (`original.bin` was
      restored; export writes only `New Account` back)

## 9. Client use

Set **Client / project name first**, then upload their file.

Switching the client name switches:

- SQLite memory scope (rules, mappings, training, account profiles)
- the per-client model folder (`/data/models/clients/<safe_id>/`)
- the session folder (`/data/sessions/<safe_id>/`)

An empty name is the shared/`""` scope and will mix books. Live mode blocks
upload and resume until a name is set.

**Unload file** clears the browser session and can delete that client's
session folder. It never touches SQLite rules or models.

**Start fresh** on live requires typing the exact client name. Even then it
deletes that client's *session files only*, not the knowledge base.

**Reset demo data** is hidden in live mode.

## Local vs this image

`streamlit run main.py` on a laptop with no special env is unchanged: no
login, no auto-wipe, data under `./data/`, sample is opt-in. This Railway
path is an add-on, not a replacement for local-first.
