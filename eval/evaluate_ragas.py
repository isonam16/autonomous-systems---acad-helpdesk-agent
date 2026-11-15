import argparse
import asyncio
import csv
import os
import uuid
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from helpdesk import config as C
from helpdesk.agents import root_agent, single_agent
from helpdesk.rag import retrieve

# Load environment variables (.env)
load_dotenv(C.ROOT / ".env")


async def generate_response(agent, question: str) -> dict:
    """Run a query through the ADK helpdesk agent and return answer and tool calls."""
    svc = InMemorySessionService()
    runner = Runner(agent=agent, app_name="ragas_eval", session_service=svc)
    sid = uuid.uuid4().hex
    await svc.create_session(app_name="ragas_eval", user_id="eval_user", session_id=sid)

    msg = types.Content(role="user", parts=[types.Part(text=question)])
    answer = ""
    tools = []

    async for ev in runner.run_async(user_id="eval_user", session_id=sid, new_message=msg):
        for fc in ev.get_function_calls() or []:
            tools.append(fc.name)
        if ev.is_final_response() and ev.content and ev.content.parts:
            answer = "".join(p.text or "" for p in ev.content.parts)

    return {"answer": answer.strip(), "tools": tools}


def get_retrieved_contexts(question: str) -> list[str]:
    """Retrieve text chunks from the RAG index for Ragas context evaluation."""
    try:
        res = retrieve(question, top_k=C.TOP_K)
        return [p["text"] for p in res.get("results", [])]
    except Exception:
        return []


async def run_evaluation(mode: str = "multi", limit: int = None, out_path: str = "eval/ragas_results.csv"):
    questions_file = Path(__file__).resolve().parent / "questions.csv"
    if not questions_file.exists():
        raise FileNotFoundError(f"Questions file not found at {questions_file}")

    with open(questions_file, encoding="utf-8-sig") as f:
        benchmarks = list(csv.DictReader(f))

    if limit and limit > 0:
        benchmarks = benchmarks[:limit]

    agent = root_agent if mode == "multi" else single_agent
    print(f"Running evaluation on {len(benchmarks)} questions (mode: {mode})...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []
    categories = []
    question_ids = []

    for item in benchmarks:
        qid = item.get("id")
        q = item.get("question", "")
        expected = item.get("expected_answer_notes", "") or item.get("expected_behavior", "")
        category = item.get("category", "")

        print(f"Processing Q{qid}: {q[:60]}...")
        result = await generate_response(agent, q)
        ctx = get_retrieved_contexts(q)

        questions.append(q)
        answers.append(result["answer"])
        contexts.append(ctx)
        ground_truths.append(expected)
        categories.append(category)
        question_ids.append(qid)

    # Construct dataset for Ragas
    data_samples = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data_samples)

    metrics = [
        faithfulness,
        answer_relevancy,
        answer_correctness,
        context_precision,
        context_recall,
    ]

    print("\nRunning Ragas evaluation metrics...")
    score_results = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    df = score_results.to_pandas()
    df.insert(0, "question_id", question_ids)
    df.insert(1, "category", categories)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False, encoding="utf-8-sig")

    print(f"\nEvaluation completed! Results saved to: {out_file}")
    print("\nSummary Metrics:")
    for metric_name, val in score_results.items():
        print(f"  {metric_name}: {val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Academic Helpdesk with Ragas")
    parser.add_argument("--mode", choices=["single", "multi"], default="multi", help="Agent mode to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Number of questions to test (default: all)")
    parser.add_argument("--out", default="eval/ragas_results.csv", help="Path to save evaluation CSV")
    args = parser.parse_args()

    asyncio.run(run_evaluation(mode=args.mode, limit=args.limit, out_path=args.out))
