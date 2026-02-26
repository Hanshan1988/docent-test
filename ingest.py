"""
Ingestion script for smolagents Langfuse trace -> Docent.
Parses trace_full.jsonl and uploads to Docent as a single AgentRun
with a full multi-turn transcript.
"""

import json
import os
import re
import ast
from pathlib import Path
from dotenv import load_dotenv

from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import (
    parse_chat_message,
    AssistantMessage,
    UserMessage,
    SystemMessage,
    ToolMessage,
    ToolCall,
)

load_dotenv()

DATA_PATH = Path(__file__).parent / "trace_full.jsonl"
COLLECTION_NAME = "smolagents-pii-dataset-task"
DOCENT_API_KEY = os.environ["DOCENT_API_KEY"]

# --- Message parsing ---

ROLE_MAP = {
    "SYSTEM": "system",
    "USER": "user",
    "ASSISTANT": "assistant",
    "TOOL_CALL": "assistant",      # tool-call messages become assistant with tool_calls
    "TOOL_RESPONSE": "tool",
}


def parse_stringified_chat_message(s: str) -> dict:
    """Parse a stringified Python ChatMessage repr into a dict with role and content."""
    # Extract role
    role_match = re.search(r"role=<MessageRole\.(\w+):", s)
    if not role_match:
        raise ValueError(f"Could not extract role from: {s[:100]}")
    role_key = role_match.group(1)
    role = ROLE_MAP.get(role_key, "user")

    # Extract content list - find the content=[...] portion
    # The content is a Python list of dicts like [{'type': 'text', 'text': '...'}]
    content_match = re.search(r"content=(\[.*\]),\s*tool_calls=", s, re.DOTALL)
    if not content_match:
        # Fallback: try to grab everything between content= and the next top-level field
        content_match = re.search(r"content=(\[.*?\])\s*[,)]", s, re.DOTALL)

    text = ""
    if content_match:
        try:
            content_list = ast.literal_eval(content_match.group(1))
            text_parts = []
            for item in content_list:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item["text"])
            text = "\n".join(text_parts)
        except (ValueError, SyntaxError):
            # Fallback: regex extract text field
            text_matches = re.findall(r"'text': '(.*?)'(?:,|\})", s)
            if text_matches:
                text = text_matches[-1]  # last 'text' field is usually the content

    return {"role_key": role_key, "role": role, "content": text}


def build_docent_messages(raw_msgs: list[str], last_gen_output: dict) -> list:
    """Convert stringified ChatMessage list + last generation output into Docent messages."""
    messages = []

    for raw in raw_msgs:
        parsed = parse_stringified_chat_message(raw)
        role = parsed["role"]
        role_key = parsed["role_key"]
        content = parsed["content"]

        if role_key == "TOOL_CALL":
            # Parse tool call info from the content text
            # Content looks like: "Calling tools:\n[{'id': 'call_1', ...}]"
            tool_calls = []
            tc_match = re.search(r"Calling tools:\n(\[.*\])", content, re.DOTALL)
            if tc_match:
                try:
                    tc_list = ast.literal_eval(tc_match.group(1))
                    for tc in tc_list:
                        func_info = tc.get("function", {})
                        raw_args = func_info.get("arguments", {})
                        # arguments must be dict; smolagents passes code as string
                        if isinstance(raw_args, str):
                            raw_args = {"code": raw_args}
                        tool_calls.append(
                            ToolCall(
                                id=tc.get("id", "unknown"),
                                function=func_info.get("name", "unknown"),
                                arguments=raw_args,
                                type="function",
                            )
                        )
                except (ValueError, SyntaxError):
                    pass

            msg = AssistantMessage(content=content, tool_calls=tool_calls or None)
            messages.append(msg)

        elif role_key == "TOOL_RESPONSE":
            # Extract tool_call_id if present
            call_id_match = re.search(r"Call id: (call_\d+)", content)
            call_id = call_id_match.group(1) if call_id_match else None
            msg = ToolMessage(
                content=content,
                tool_call_id=call_id or "unknown",
            )
            messages.append(msg)

        else:
            msg = parse_chat_message({"role": role, "content": content})
            messages.append(msg)

    # Append the last generation's output as the final assistant message
    if isinstance(last_gen_output, dict) and last_gen_output.get("content"):
        msg = AssistantMessage(content=last_gen_output["content"])
        messages.append(msg)

    return messages


# --- Main ---

def main():
    # Load trace
    with open(DATA_PATH, "r") as f:
        trace = json.loads(f.readline())

    print(f"Loaded trace: {trace['id']}")
    print(f"Task: {trace['name']}")

    # Get observations
    observations = trace["observations"]
    gens = sorted(
        [obs for obs in observations if obs["type"] == "GENERATION"],
        key=lambda x: x["startTime"],
    )
    steps = [obs for obs in observations if obs["type"] == "CHAIN"]
    tools = [obs for obs in observations if obs["type"] == "TOOL"]

    print(f"Observations: {len(observations)} total ({len(gens)} generations, {len(steps)} steps, {len(tools)} tools)")

    # Get the last generation's input messages (full conversation) + output
    last_gen = gens[-1]
    raw_msgs = last_gen["input"]["messages"]
    last_gen_output = last_gen["output"]

    print(f"Reconstructing transcript from {len(raw_msgs)} accumulated messages + 1 final output")

    # Build Docent messages
    messages = build_docent_messages(raw_msgs, last_gen_output)
    print(f"Built {len(messages)} Docent messages")

    # Compute aggregate token counts
    total_prompt_tokens = sum(g.get("promptTokens") or 0 for g in gens)
    total_completion_tokens = sum(g.get("completionTokens") or 0 for g in gens)

    # Extract metadata from trace
    input_data = trace.get("input", {})
    if isinstance(input_data, str):
        input_data = json.loads(input_data)
    task = input_data.get("task", "")

    meta_attrs = trace.get("metadata", {}).get("attributes", {})
    if isinstance(meta_attrs, str):
        meta_attrs = json.loads(meta_attrs)

    tools_names = meta_attrs.get("smolagents.tools_names", "[]")
    if isinstance(tools_names, str):
        try:
            tools_names = json.loads(tools_names)
        except json.JSONDecodeError:
            tools_names = ast.literal_eval(tools_names)
    max_steps = meta_attrs.get("smolagents.max_steps", "")

    # Build AgentRun
    transcript = Transcript(messages=messages)

    agent_run = AgentRun(
        transcripts=[transcript],
        metadata={
            "trace_id": trace["id"],
            "timestamp": trace["timestamp"],
            "agent_name": trace["name"],
            "task": task,
            "tools": tools_names,
            "max_steps": int(max_steps) if max_steps else None,
            "num_steps": len(steps),
            "model": gens[0].get("model") if gens else None,
            "scores": {
                "latency_seconds": trace.get("latency"),
                "total_cost": trace.get("totalCost"),
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
            },
        },
    )

    # Quick validation - ensure messages were built
    assert len(messages) > 0, "No messages built"
    assert len(agent_run.transcripts) == 1, "Expected 1 transcript"
    print(f"Validation passed: {len(messages)} messages in 1 transcript")

    # Upload to Docent
    client = Docent(api_key=DOCENT_API_KEY)

    collection_id = client.create_collection(
        name=COLLECTION_NAME,
        description="smolagents CodeAgent traces for PII dataset discovery task. "
        "Used for failure mode analysis and comparison across runs.",
    )
    print(f"Created collection: {collection_id}")

    client.add_agent_runs(collection_id, [agent_run])
    print(f"Uploaded 1 agent run")
    print(f"View at: https://docent.transluce.org/collection/{collection_id}")

    # Return for verification
    return collection_id, agent_run


if __name__ == "__main__":
    result = main()
