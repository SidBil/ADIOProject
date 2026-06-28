# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

This is a research-plus-product monorepo for **ADI/O**, a multimodal ASR therapy app for autism-focused reading comprehension. There are four largely independent top-level areas:

- `asr/` — Whisper + LoRA fine-tuning research (Jupyter notebooks, `config.yaml`, checkpoint dir `finetuned-whsiper-lora/`). Notebooks run against the TORGO dataset under `audio/`.
- `imagegen/` — Tools to generate and curate the 100-image bank and its metadata (`generate_images.py`, `generate_metadata.py`, `image_metadata.csv`).
- `multimodal/` — Notebooks for CLIP-based decoder biasing experiments. Produces `web/data/cache/clip_image_embeddings.npz`, which is consumed by the production ASR endpoint.
- `web/` — The production app: an Expo/React Native (web target) frontend, a FastAPI backend, a Modal-hosted GPU inference service, and a Supabase database. **`webV1/` is a legacy version — do not edit unless explicitly asked.**

The root-level `requirements.txt` and `venv/` are for the research notebooks; `web/` has its own `requirements.txt` and `venv/`.

## web/ — How the three runtimes fit together

The web app has three deployment targets that must be reasoned about together:

1. **Frontend (`App.tsx` + `src/`)** — Expo React Native targeting web. Built with `npx expo export --platform web` to `dist/`. Auth is Supabase; on `localhost` it auto-signs-in as `dev@adio.local` so OAuth redirect isn't needed in dev. Screen flow is a flat string-state machine in `App.tsx` (`landing → welcome → session → summary`, plus `onboarding`, `dashboard`, `about`).

2. **Backend (`app.py` + `services/`)** — FastAPI. Locally run with `python app.py` (uvicorn on `:8000`, hit from frontend via `src/config.ts`). In prod it's wrapped by `api/index.py` as a Vercel serverless function (see `vercel.json` rewrites for `/api/*` and `/images/*`). **Sessions are kept in an in-memory dict on `SessionManager`** — on Vercel's cold-started serverless instances this dict is empty, so `SessionManager.recover_session()` reconstructs sessions from Supabase rows. Any new endpoint that needs session state must fall back to `sessions.get_session(...) or sessions.recover_session(...)`.

3. **GPU inference (`modal_asr.py`)** — A Modal app (`adio-asr`) that hosts Whisper-small + LoRA (`zorbbbb/whisper-small-lora-torgo`) merged with CLIP rescoring. Deploy with `modal deploy web/modal_asr.py`; iterate with `modal serve web/modal_asr.py`. The backend reaches it via `MODAL_ASR_URL` (transcribe) and `MODAL_WARMUP_URL` (ping container). When `MODAL_ASR_URL` is unset, `ASRService` falls back to loading models locally from `asr/finetuned-whsiper-lora/` — slow on CPU/MPS but useful for offline work.

**Session start blocks on Modal warmup**: `start_session` runs the DB insert and `asr.warmup()` in parallel via `asyncio.gather` to ensure the GPU container is loaded before the user records their first answer. Don't change this to fire-and-forget without understanding the latency tradeoff.

### LLM evaluation flow

`LLMService` (GPT-4o) is called only at evaluation time (and only for follow-up when `accuracy < ACCURACY_THRESHOLD = 4`). Session start does **not** call the LLM — questions come from the pre-generated `data/image_metadata.csv` (10 structure-word Q/A pairs per image: who/what/where/color/shape/sound/size/number/movement/mood). Adding fields to `image_metadata.csv` requires updating `_load_metadata()` in `session_manager.py`.

### Multimodal rescoring

Both `modal_asr.py` and the local-fallback `services/asr_service.py` implement the same `_rescore` pipeline: Whisper beam search → CLIP-text-embed each hypothesis → cosine similarity vs cached image embedding → softmax fusion with `alpha=0.3` (CLIP weight). The image embeddings cache (`web/data/cache/clip_image_embeddings.npz`) is **baked into the Modal image at build time** via `add_local_file(...)`, so adding new images requires both regenerating the cache and redeploying Modal.

### Supabase / data layer

`services/interaction_store.py` is the single boundary to Supabase. Schema is defined by `web/supabase/migrations/001_instrumentation.sql` and `002_storage_policies.sql`; tables: `user_profiles`, `user_consents`, `therapy_sessions`, `therapy_turns`, `analytics_errors`, plus the private `therapy-audio` storage bucket. RLS is on — every table scopes by `auth.uid()`, and all backend DB calls use the **caller's JWT**, not a service key. The backend extracts the JWT via the `get_token` dependency in `app.py`. See `DATABASE_SCHEMA_SUMMARY.md` for the full rationale.

## Commands

### Backend (web/)

```bash
cd web
python app.py                                # local FastAPI on :8000
modal deploy web/modal_asr.py                # deploy GPU inference
modal serve web/modal_asr.py                 # hot-reload Modal dev
python test_storage_upload.py                # smoke-test Supabase storage
```

Requires `web/.env` with at minimum `OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and (for prod-equivalent local) `MODAL_ASR_URL` + `MODAL_WARMUP_URL`.

### Frontend (web/)

```bash
cd web
npm install
npm run web                                  # Expo on :8081 (web target)
npm run ios | npm run android
npm run build:web                            # → dist/ (what Vercel serves)
```

### Research notebooks (asr/, multimodal/, imagegen/)

```bash
source venv/bin/activate                     # root-level venv
pip install -r requirements.txt
jupyter notebook
```

Fine-tuning config lives in `asr/config.yaml`. The notebooks expect to be run from the project root so paths like `audio/torgo/processed/` resolve.

## Conventions worth knowing

- **Image filenames are IDs.** `image_id` and `image_filename` are typically the same `img_XXX.png` string; `SessionManager` keys metadata by filename.
- **ASR transcriptions have no caps or punctuation** — the LLM evaluation prompt is explicit about this, so don't add post-processing that "fixes" them; it will desync the eval prompt's expectations.
- **Two `webV1/` and `web/` directories exist**; `webV1/` is the previous Flask/Jinja version and is no longer wired up. New work goes in `web/`.
- **The LoRA path has a typo** (`asr/finetuned-whsiper-lora/`) — preserve it; both training and serving code reference that exact path.
