"""FastAPI API for the IIT Jodhpur Academic Helpdesk."""
import os,uuid,logging
from typing import Optional
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from helpdesk import config as C
from helpdesk.agents import root_agent,single_agent
APP="helpdesk_api"
app=FastAPI(title="IIT Jodhpur Academic Helpdesk API")
origins=[x.strip() for x in os.getenv("FRONTEND_ORIGIN","").split(",") if x.strip()]
app.add_middleware(CORSMiddleware,allow_origins=origins+["http://localhost:3000"],allow_origin_regex=r"https://([a-z0-9-]+\\.)*vercel\\.app",allow_credentials=True,allow_methods=["GET","POST"],allow_headers=["*"])
sessions=InMemorySessionService()
runners={"multi":Runner(agent=root_agent,app_name=APP,session_service=sessions),"single":Runner(agent=single_agent,app_name=APP,session_service=sessions)}
class ChatRequest(BaseModel):
 message:str
 session_id:Optional[str]=None
 mode:str="multi"
class ChatResponse(BaseModel):
 answer:str
 session_id:str
 agent:str=""
 tools:list[str]=Field(default_factory=list)
@app.get("/health")
def health(): return {"status":"ok","model":C.LLM_MODEL,"embedding_model":C.EMBED_MODEL}
@app.post("/chat",response_model=ChatResponse)
async def chat(req:ChatRequest):
 message=req.message.strip()
 if not message: raise HTTPException(400,"message cannot be empty")
 if req.mode not in runners: raise HTTPException(400,"mode must be 'multi' or 'single'")
 sid=req.session_id or uuid.uuid4().hex
 if not await sessions.get_session(app_name=APP,user_id="web",session_id=sid): await sessions.create_session(app_name=APP,user_id="web",session_id=sid)
 content=types.Content(role="user",parts=[types.Part(text=message)])
 answer,agent,called_tools="","",[]
 try:
  async for ev in runners[req.mode].run_async(user_id="web",session_id=sid,new_message=content):
   called_tools.extend(fc.name for fc in (ev.get_function_calls() or []))
   if ev.is_final_response() and ev.content and ev.content.parts:
    value="".join(p.text or "" for p in ev.content.parts).strip()
    if value: answer,agent=value,ev.author
 except Exception:
  logging.exception("Helpdesk request failed")
  raise HTTPException(500,"The helpdesk could not process this request.")
 return ChatResponse(answer=answer or "I could not generate a response. Please try again.",session_id=sid,agent=agent,tools=called_tools)
