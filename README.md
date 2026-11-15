# IIT Jodhpur Academic Helpdesk

Multi-agent academic helpdesk using Google ADK, Gemini API, ChromaDB and BM25. The Next.js frontend is deployed on Vercel; the FastAPI backend and persistent ChromaDB index run on Render.

## Architecture
- Frontend: Next.js App Router + TypeScript + Tailwind CSS (`frontend/`)
- API: FastAPI + Google ADK (`backend/`)
- Chat model: Gemini API, default `gemini-2.5-flash`
- Embeddings: Gemini API, default `gemini-embedding-001`
- Retrieval: persistent ChromaDB + BM25 + Reciprocal Rank Fusion
- Existing policy, schedule, email and orchestrator agents retained

## Deploy backend to Render
1. Connect this repository to Render and create a Blueprint from `render.yaml`.
2. Select branch `deployment/vercel-render-gemini`.
3. Set `GEMINI_API_KEY` in Render environment variables (https://aistudio.google.com/apikey).
4. Keep the persistent disk mounted at `/var/data`. The first start indexes the PDFs in `data/docs/` using Gemini embeddings. Later restarts reuse that index.
5. Verify `https://YOUR-RENDER-SERVICE.onrender.com/health`.

The Render service uses a persistent disk, so a paid Render plan with disk support is required. Initial indexing can take several minutes and uses Gemini embedding API quota.

## Deploy frontend to Vercel
1. Import this repository into Vercel.
2. Set Root Directory to `frontend`.
3. Set Production Branch to `deployment/vercel-render-gemini`.
4. Add `NEXT_PUBLIC_API_URL` with your Render backend URL (no trailing slash).
5. Deploy.

For a custom frontend domain, set `FRONTEND_ORIGIN` on Render to its exact origin.

## Local development
Backend from the repository root:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add GEMINI_API_KEY to .env
python -m helpdesk.ingest
uvicorn backend.main:app --reload --port 8000
```
Frontend:
```bash
cd frontend
npm install
echo 'NEXT_PUBLIC_API_URL=http://localhost:8000' > .env.local
npm run dev
```
Open http://localhost:3000.

## Dataset note
Document search uses PDFs/text files in `data/docs/`. Course timetable and instructor email tools also require `data/timetable.csv` and `data/faculty.csv`; add these two CSVs to enable those features.

## Environment variables
- `GEMINI_API_KEY`: required API key
- `GEMINI_MODEL`: chat model (default `gemini-2.5-flash`)
- `EMBED_MODEL`: embedding model (default `gemini-embedding-001`)
- `CHROMA_DIR`, `CHUNKS_JSON`: persistent index paths (set by Render)
- `FRONTEND_ORIGIN`: optional comma-separated custom frontend origins

No Ollama service is required by this deployment.
