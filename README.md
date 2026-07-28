# Intelligent Call Center Automation Platform

## Overview

Intelligent Call Center Automation Platform is an AI-powered multi-agent system designed to augment call center workflows with real-time voice capture, transcription, and automation-ready infrastructure. The platform is being built to support live audio ingestion, speech-to-text processing, and future agentic workflows that can assist with call handling, summarization, and operational support.

## Tech Stack

- FastAPI
- React
- TypeScript
- Tailwind CSS
- Docker
- PostgreSQL
- Redis
- n8n
- Groq Whisper
- LangGraph, planned for Phase 2+

## Current Status

Phase 1 foundation is complete. The current implementation includes:

- Docker infrastructure for local development services
- FastAPI backend with health checks
- WebRTC audio capture in the frontend
- Real-time WebSocket transcription pipeline powered by Groq Whisper

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

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

## Known Limitations

- Transcription accuracy depends heavily on microphone quality and background noise.
- Chunked audio transcription can introduce brief context loss between 3-second segments.

## Roadmap

- Phase 1 (Complete): Foundation — Docker infrastructure, FastAPI backend, WebRTC audio capture, real-time WebSocket transcription via Groq Whisper
- Phase 2: Core Intelligence Agents — Intent Classification Agent (DistilBERT), LangGraph orchestration, Knowledge Retrieval Agent (RAG over ChromaDB)
- Phase 3: Agent-Assist Experience — Response Suggestion Agent (Groq Llama 3.3 70B), live WebSocket suggestions, Escalation Prediction Agent (XGBoost)
- Phase 4: Quality & Automation — Quality Assurance Agent with rubric scoring, After-Call Work Agent, n8n CRM/ticketing integration
- Phase 5: Hardening & Evaluation — Load testing, bias/fairness evaluation, security review (PII redaction, RBAC)
- Phase 6: Pilot Deployment — Limited pilot, baseline vs. pilot metrics (AHT, CSAT, FCR), production upgrade path documentation
