
"""Gradio frontend, mounted inside the FastAPI application."""

import os
import uuid

import gradio as gr
import requests

# On Render, use the same server's port.
# Locally, default to the FastAPI development port 8000.
API = os.getenv("BACKEND_URL") or (
    f"http://127.0.0.1:{os.getenv('PORT', '8000')}"
)


def respond(message, history, mode, session_id):
    try:
        r = requests.post(
            f"{API}/chat",
            timeout=300,
            json={
                "message": message,
                "session_id": session_id,
                "mode": mode,
            },
        )
        r.raise_for_status()
        d = r.json()

        return (
            f"{d['answer']}\n\n"
            f"_agent: {d['agent']} | "
            f"tools: {', '.join(d['tools']) or 'none'}_"
        )

    except Exception as e:
        return f"Error contacting backend: {e}"


with gr.Blocks(title="IIT Jodhpur Academic Helpdesk") as demo:
    gr.Markdown("# IIT Jodhpur Academic Helpdesk")

    mode = gr.Radio(
        ["multi", "single"],
        value="multi",
        label="Agent mode",
    )

    sid = gr.State(lambda: uuid.uuid4().hex)

    gr.ChatInterface(
        respond,
        additional_inputs=[mode, sid],
        type="messages",
    )
