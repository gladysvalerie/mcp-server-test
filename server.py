import os
import json
import re
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

load_dotenv()

client = OpenAI()

mcp = FastMCP(
    name="Coding TA",
    port=8000
)

@mcp.tool()
def list_problem() -> str:
    """ Return the full list of problems as JSON. """
    try:
        problems_path = os.path.join(os.path.dirname(__file__), "data", "problems.json")

        with open(problems_path, "r") as f:
            problems = json.load(f)

        return json.dumps({"ok": True, "problems": problems})

    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


def safe(data: str):
    """Normalize AI output into proper JSON with minimal cleanup."""

    if not isinstance(data, str):
        return json.dumps({"ok": False, "error": "Expected string", "raw": data})

    text = data.strip()

    # Strip code fences like ```json ... ```
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "")
        text = text.strip()

    # Detect JSON object body without braces
    # Example: "  \"is_correct\": true, \"score\": 90 "
    if re.match(r'^\s*\"[A-Za-z0-9_]+\"\s*:', text) or re.match(r'^\s*\n*\s*\"[A-Za-z0-9_]+\"\s*:', text):
        text = "{" + text.lstrip() + "}"
    print(text)

    # Parse JSON
    try:
        parsed = json.loads(text)
        return json.dumps({"ok": True, "data": parsed})
    except Exception:
        return json.dumps({"ok": False, "error": "Invalid JSON", "raw": data})

    
@mcp.tool()
def generate_subproblems(problem_id: int) -> str:
    """
    Generate 3–6 subproblems for the given problem ID.
    Always use this tool when the user asks for subproblems, decomposition, breakdown, or steps.
    """
    try:
        problems_path = os.path.join(os.path.dirname(__file__), "data", "problems.json")
        with open(problems_path, "r") as f:
            problems = json.load(f)

        problem = next((p for p in problems if p["id"] == problem_id), None)
        if not problem:
            return f"Error: problem_id {problem_id} not found"

        prompt = f"""
        Break this problem into subproblems.
        Return 3–6 subproblems max.

        Format the subproblems as JSON list:

        [
          {{"title": "...", "description": "..."}},
          ...
        ]

        Problem:
        Title: {problem['title']}
        Description: {problem['description']}
        """

        ai_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        content = ai_resp.choices[0].message.content

        return safe(content)

    except Exception as e:
        return f"Error: {e}"
    
@mcp.tool()
def get_problems_by_topic(topic: str | list):
    problems_path = os.path.join(os.path.dirname(__file__), "data", "problems.json")

    with open(problems_path, "r") as f:
        problems = json.load(f)

    if isinstance(topic, str):
        topics = [topic.lower()]
    elif isinstance(topic, list):
        topics = [t.lower() for t in topic if isinstance(t, str)]
        if not topics:
            return {"ok": False, "error": "topic list contains no valid strings"}
    else:
        return {"ok": False, "error": "topic must be a string or list of strings"}

    results = []

    for p in problems:
        raw = p.get("topic", [])

        if isinstance(raw, str):
            problem_topics = [raw.lower()]
        elif isinstance(raw, list):
            problem_topics = [t.lower() for t in raw if isinstance(t, str)]
        else:
            problem_topics = []

        # Check match
        if any(t in pt for t in topics for pt in problem_topics):
            results.append(p)

    return {"ok": True, "results": results}

@mcp.tool()
def check_solution(problem_id: int, code: str) -> str :
    content = None
    """
    Check a student's solution for a given problem.

    This does LLM-based evaluation only (no code execution).
    Returns JSON with correctness, score, and feedback.
    """
    try:
        problems_path = os.path.join(os.path.dirname(__file__), "data", "problems.json")
        with open(problems_path, "r") as f:
            problems = json.load(f)

        problem = next((p for p in problems if p["id"] == problem_id), None)
        if not problem:
            return json.dumps({"ok": False, "error": f"problem_id {problem_id} not found"})

        prompt = """
Return ONLY valid JSON. No backticks. No explanation.

Required format:
{{
  "is_correct": true,
  "score": 0-100,
  "feedback": "string",
  "issues": ["string", ...]
}}

Evaluate this solution.

Problem:
{title}

Description:
{description}

Student code:
{code}
    """.format(
        title=problem["title"],
        description=problem["description"],
        code=code,
        )
        ai_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        content = ai_resp.choices[0].message.content
        print(content)
        # Normalize to JSON-ish structure using your existing helper
        return safe(content)

    except Exception as e:
        return json.dumps({"ok": False, "error": str(e), "raw": content})
    
if __name__ == "__main__":
    mcp.run(transport="stdio") 