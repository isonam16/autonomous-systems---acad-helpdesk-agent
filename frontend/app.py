
"""Gradio frontend, mounted by the FastAPI backend."""

import uuid

import gradio as gr

async def respond(message, history, mode, session_id):
    # Call the backend logic in-process: no HTTP to localhost (fails on Vercel).
    try:
        from backend.main import ChatRequest, run_chat

        d = await run_chat(
            ChatRequest(message=message, session_id=session_id, mode=mode)
        )
        return (
            f"{d.answer}\n\n"
            f"_agent: {d.agent} | tools: {', '.join(d.tools) or 'none'}_"
        )
    except Exception as e:
        return f"Error: {getattr(e, 'detail', e)}"


with gr.Blocks(title="IIT Jodhpur Academic Helpdesk") as demo:

    gr.Markdown("# IIT Jodhpur Academic Helpdesk")

    mode = gr.Radio(
        ["multi", "single"],
        value="multi",
        label="Agent mode",
    )

    sid = gr.State(lambda: uuid.uuid4().hex)

    gr.ChatInterface(
        fn=respond,
        additional_inputs=[mode, sid],
    )
