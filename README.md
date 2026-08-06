# Intelligent Call Center Automation Platform

## Overview

Intelligent Call Center Automation Platform is an AI-powered multi-agent system designed to augment call center workflows with real-time voice capture, transcription, and agentic automation. The platform ingests live audio, transcribes it in real time, and runs a multi-agent LangGraph pipeline that classifies caller intent, retrieves grounded knowledge-base content, drafts response suggestions, and predicts escalation risk. Once a call ends, two additional agents score the call against a QA rubric and generate a structured after-call summary with follow-up tasks — all on a $0 infrastructure stack using free-tier and self-hosted open-source tools.

This is a portfolio project built to demonstrate production-quality agentic AI system design: multi-agent orchestration, RAG, fine-tuned local ML models, and a documented upgrade path from a zero-cost prototype to a production deployment.

## Tech Stack

**Backend & Orchestration**
- FastAPI, LangGraph
- PostgreSQL, Redis, n8n
- Docker / Docker Compose

**Frontend**
- React, TypeScript, Tailwind CSS

**AI / ML**
- Groq Whisper Large v3 Turbo — real-time speech-to-text (with faster-whisper offline fallback)
- Groq Llama 3.3 70B — grounded response generation, QA rubric scoring, and after-call summarization (with Gemini/Ollama fallback via a provider-agnostic `LLMClient` wrapper)
- Fine-tuned DistilBERT — local intent classification (8-category taxonomy)
- ChromaDB + sentence-transformers (`all-MiniLM-L6-v2`) — local embedded vector search
- rank-bm25 — keyword search, fused with semantic search via Reciprocal Rank Fusion
- XGBoost — local escalation-risk classifier
- VADER (`vaderSentiment`) — local, lexicon-based sentiment trend tracking

## Current Status

**Phase 1 (Foundation) — Complete.** Docker infrastructure, FastAPI backend, WebRTC audio capture, real-time WebSocket transcription via Groq Whisper.

**Phase 2 (Core Intelligence Agents) — Complete.**
- Intent Classification Agent — fine-tuned DistilBERT, 90.62% validation accuracy (weighted F1 0.90), empirically calibrated confidence threshold of 0.35
- Knowledge Retrieval Agent — hybrid RAG over ChromaDB (semantic + BM25, fused via RRF)
- LangGraph orchestration graph established (`transcribe → classify_intent → retrieve_knowledge`)

**Phase 3 (Agent-Assist Experience) — Complete.**
- Response Suggestion Agent — Groq Llama 3.3 70B, strictly grounded in retrieved KB content, with safe fallback behavior when no KB match exists
- Escalation Prediction Agent — locally trained XGBoost classifier, **AUC-ROC 0.86** (target: >0.80), calibrated alert threshold of **0.4** (false-positive rate ≈13%, target: <15%)
- Sentiment tracking service (VADER-based) feeding sentiment trend, current sentiment, and worst-sentiment-so-far into escalation prediction
- Full graph wiring: `classify_intent` now branches to `retrieve_knowledge → suggest_response` (sequential, since suggestions require retrieved KB content) and to `predict_escalation` (parallel, since it only depends on intent and sentiment)
- Live dashboard (`CallDashboard.tsx`) wired end-to-end over WebSocket (`/ws/calls/{call_id}`), backed by Redis session state (1-hour TTL) so per-turn graph invocations share state across a call

**Phase 4 (Quality & Automation) — Complete.**
- Quality Assurance Agent — Groq Llama 3.3 70B scores the completed call transcript against a fixed 6-dimension rubric (`greeting`, `active_listening`, `compliance`, `resolution`, `tone`, `closing`), each 0–100, plus an independently-generated overall score, transcript-grounded coaching flags, and a rationale that must reference a specific moment or phrase from the actual call — not generic filler. Exposed as `GET /calls/{call_id}/qa_score`, deliberately **not** a LangGraph node (see *Design Decisions* below), and cached in Redis so downstream agents can reuse it.
- After-Call Work Agent — Groq Llama 3.3 70B generates a structured summary (resolution status, commitments made, compliance-relevant statements, follow-up tasks with priority/deadline, and a `requires_human_review` flag with reason for low-confidence transcripts) and triggers a non-blocking n8n webhook with the result. Exposed as `GET /calls/{call_id}/after_call_summary`, reads the cached QA score when available rather than re-scoring.
- Both agents surfaced in the dashboard as manually-triggered post-call panels ("Get QA Score" / "Get Call Summary"), rather than auto-firing on call end — avoids an LLM call on every test-call during development and mirrors a real supervisor's on-demand review workflow.
- Shared JSON-fence-stripping logic deduped into `services/llm_utils.py` after being independently duplicated across `quality.py` and `after_call.py`.
- CORS middleware added to the backend — the QA/after-call panels were the first features to call the REST API directly from the browser (all prior communication was WebSocket, which isn't subject to CORS), which surfaced the gap.

## Design Decisions

A few deliberate deviations from a naive reading of the spec, made for good reasons:

- **QA and After-Call Work are not LangGraph nodes.** The graph is invoked once per audio turn; both agents need the *complete* transcript to produce meaningful output (you can't score "closing" quality or summarize resolution status mid-call). They're exposed as on-demand REST endpoints instead, triggered after the call ends.
- **QA's `overall_score` is generated independently from the 6 dimension scores**, not computed as their average. This lets the model weigh a serious single issue (e.g. a compliance violation) disproportionately, the way a real QA reviewer would, rather than diluting it into a flat average.
- **After-Call Work's `requires_human_review` flag is independent from `resolution_status`.** A call can be marked `"resolved"` and still be flagged for human review if the transcript is too short or ambiguous to summarize with full confidence — these are two different questions (what happened vs. how confident are we in this summary), and collapsing them into one would lose a real signal.
- **The n8n integration is a demonstrable workflow-automation pattern, not a real CRM.** No production CRM/ticketing system exists for this project; the webhook proves the architectural pattern (agent output → automated downstream workflow trigger) rather than claiming a live Salesforce/Zendesk integration that doesn't exist.

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd call-center-automation
```

### 2. Start the local infrastructure

From the `docker/` folder, start the supporting services:

```bash
cd docker
docker compose up -d
```

This brings up PostgreSQL, Redis, and n8n.

### 3. Start the backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 4. Create backend/.env

Create a `.env` file inside `backend/` with the following variables:

```env
DATABASE_URL=postgresql+psycopg2://calluser:callpassword@localhost:5433/callcenter
REDIS_URL=redis://localhost:6379/0
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PERSIST_DIR=./data/chroma
```

A free Groq API key can be obtained at [console.groq.com](https://console.groq.com) under [API Keys](https://console.groq.com/keys).

`DEFAULT_AGENT_ID` and `N8N_WEBHOOK_URL` have working defaults in `config.py` and don't need to be set unless you're pointing at a different n8n instance.

### 5. Regenerate local model artifacts (gitignored, not committed)

Trained model weights are excluded from version control and must be regenerated locally:

```bash
# Intent classifier (requires a GPU — trained via Colab; see backend/fine_tuning/)
python -m fine_tuning.train_intent_classifier

# Escalation classifier (lightweight, trains locally on CPU in seconds)
python -m fine_tuning.prepare_escalation_data
python -m fine_tuning.train_escalation_classifier

# Knowledge base index
python -m scripts.index_kb
```

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at `http://localhost:5173`; the backend's CORS policy is scoped to allow requests from that origin specifically.

### 7. Take a call

Open the dashboard, click **Start Call**, and talk. Live transcript, intent, KB matches, suggested responses, and escalation risk update in real time. Click **End Call**, then use **Get QA Score** and **Get Call Summary** to trigger the post-call agents.

## Known Limitations

- **Transcription accuracy** depends heavily on microphone quality and background noise; chunked audio transcription can introduce brief context loss between 3-second segments.
- **Intent classifier confidence is under-calibrated.** The confidence threshold was empirically set to 0.35 (rather than a conventional 0.6–0.8) because the small training dataset (640 examples, 4 epochs) produces genuine softmax under-confidence even on correct, in-distribution predictions. Argmax accuracy (90.62%) is unaffected; only the confidence *magnitude* is low. A larger training set at scale would likely allow a more conventional threshold — this is a deliberate, documented tradeoff, not an oversight.
- **VADER sentiment scoring struggles with negation-heavy phrasing.** As a lexicon/rule-based tool (not a trained language model), VADER can misjudge sentences where a negation word doesn't sit adjacent to the sentiment-bearing word it's meant to negate. For example, "Still not fixed, very annoyed" scores as mildly *positive* (compound +0.34), because "still" reads as mildly positive and "not" is parsed as negating "fixed" rather than scoping over "annoyed." This is a known, general limitation of lexicon-based sentiment analysis, not specific to this implementation. A production system facing this at scale might use a lightweight fine-tuned transformer for sentiment instead, at the cost of higher latency and compute.
- **The sentiment trend feature can lose signal on non-monotonic patterns.** Trend is computed as a linear regression slope over the last 5 sentiment scores. A call that dips negative, briefly returns to neutral, then drops negative again can produce a near-zero slope despite a clearly deteriorating conversation, since a straight-line fit only captures net start-to-end direction. A recency-weighted or windowed min/max feature would better capture this shape; documented here as a known direction for improvement rather than implemented, to keep the feature set consistent with what the escalation model was actually trained and evaluated against.
- **Escalation Prediction is trained entirely on synthetic, heuristically-labeled data** — no real call center data exists for this project. The labeling logic (in `fine_tuning/prepare_escalation_data.py`) is a transparent, documented, reviewable heuristic combining intent confidence, sentiment trend/minimum, call length, and a latency-based hold-time proxy, with an odds-power sharpening transform applied to keep the label distribution learnable (raising achievable AUC-ROC from a measured ceiling of 0.73 to 0.88 before jitter). This is disclosed here deliberately: the 0.86 AUC-ROC reflects how well XGBoost recovers a designed synthetic signal, not real-world escalation prediction accuracy.
- **`hold_time_proxy` is a substitution, not real telephony data.** The zero-cost stack has no real hold-time/telephony integration, so hold time is approximated as the wall-clock latency between transcript segments. This is documented as an explicit engineering tradeoff with a clear production upgrade path (real telephony hold-state), not a hidden gap.
- **QA scoring's target correlation metric (>0.85 Pearson vs. human review) is unvalidated.** No human-labeled QA dataset exists for this project, so this target from the original spec is aspirational and would need real human-reviewed calls to actually measure — not something this build can verify locally.
- **The n8n/CRM integration is simulated, not a production system.** The webhook fires against a local n8n instance to demonstrate the workflow-trigger pattern; there is no real downstream CRM or ticketing system receiving these updates.

## Roadmap

- **Phase 1 (Complete):** Foundation — Docker infrastructure, FastAPI backend, WebRTC audio capture, real-time WebSocket transcription via Groq Whisper
- **Phase 2 (Complete):** Core Intelligence Agents — Intent Classification Agent (DistilBERT), LangGraph orchestration, Knowledge Retrieval Agent (RAG over ChromaDB)
- **Phase 3 (Complete):** Agent-Assist Experience — Response Suggestion Agent (Groq Llama 3.3 70B), Escalation Prediction Agent (XGBoost + VADER sentiment tracking), full graph wiring across both agents, live dashboard wired end-to-end over WebSocket with Redis session state
- **Phase 4 (Complete):** Quality & Automation — Quality Assurance Agent with rubric scoring, After-Call Work Agent with summarization and n8n workflow triggers, both surfaced in the dashboard
- **Phase 5:** Hardening & Evaluation — Load testing, bias/fairness evaluation, security review (PII redaction, RBAC)
- **Phase 6:** Pilot Deployment — Limited pilot, baseline vs. pilot metrics (AHT, CSAT, FCR), production upgrade path documentation