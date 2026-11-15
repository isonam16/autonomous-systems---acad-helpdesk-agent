
"""FastAPI backend with mounted Gradio frontend.

Run from repository root:
uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import uuid
from typing import Optional

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from helpdesk import config as C
from helpdesk.agents import root_agent, single_agent


APP = "helpdesk_api"

# FastAPI application
app = FastAPI(title="Academic Helpdesk API")

# Session service
svc = InMemorySessionService()

# Agent runners
runners = {
    "multi": Runner(
        agent=root_agent,
        app_name=APP,
        session_service=svc,
    ),
    "single": Runner(
        agent=single_agent,
        app_name=APP,
        session_service=svc,
    ),
}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    mode: str = "multi"


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    agent: str = ""
    tools: list[str] = []


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": C.LLM_MODEL,
    }


@app.on_event("startup")
async def _startup():
    # chunk + embed docs only once (skipped if index already exists)
    from helpdesk.ingest import ensure_ingested
    await asyncio.to_thread(ensure_ingested)


async def run_chat(req: ChatRequest) -> ChatResponse:
    if req.mode not in runners:
        raise HTTPException(
            status_code=400,
            detail="mode must be 'multi' or 'single'",
        )

    sid = req.session_id or uuid.uuid4().hex

    if not await svc.get_session(
        app_name=APP,
        user_id="web",
        session_id=sid,
    ):
        await svc.create_session(
            app_name=APP,
            user_id="web",
            session_id=sid,
        )

    msg = types.Content(
        role="user",
        parts=[types.Part(text=req.message)],
    )

    answer = ""
    agent = ""
    tools = []

    try:
        async for ev in runners[req.mode].run_async(
            user_id="web",
            session_id=sid,
            new_message=msg,
        ):
            tools += [
                fc.name
                for fc in ev.get_function_calls() or []
            ]

            if (
                ev.is_final_response()
                and ev.content
                and ev.content.parts
            ):
                text = "".join(
                    p.text or ""
                    for p in ev.content.parts
                ).strip()

                if text:
                    answer = text
                    agent = ev.author

    except Exception as e:
        import logging
        logging.exception("Helpdesk request failed")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    return ChatResponse(
        answer=answer,
        session_id=sid,
        agent=agent,
        tools=tools,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    return await run_chat(req)


# --------------------------------------------------
# Mount Gradio frontend
# --------------------------------------------------

from frontend.app import demo

app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
)
