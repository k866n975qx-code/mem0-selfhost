import os
import textwrap

from dotenv import load_dotenv
from openai import OpenAI

from app.mem0_client import Mem0Client

# Load env from .env at project root
load_dotenv()

# ---- Global state ----

# OpenAI + Mem0 clients
oa_client = OpenAI()
# We'll pass agent_id explicitly so we can swap agents at runtime.
mem_client = Mem0Client(default_user_id="jose", default_agent_id=None)

CURRENT_AGENT_ID = "vera"


SYSTEM_PROMPT = """You are Jose's personal assistant.
You have access to an external long-term memory system.
Use the provided memories as context, but do NOT mention the memory system by name.
Always keep answers short, direct, and practical.
"""


# ---------------- Memory formatting helpers ----------------

def _normalize_mem_items(obj):
    """Normalize Mem0 responses into a list of memory-like dicts."""
    if obj is None:
        return []

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        # common pattern: {"results": [ ... ]}
        if "results" in obj and isinstance(obj["results"], list):
            return obj["results"]
        # fallback: treat the dict itself as a single item
        return [obj]

    # fallback: unknown type, wrap
    return [obj]


def format_memories(memories, limit: int = 20) -> str:
    """Turn raw memories into a readable list."""
    items = _normalize_mem_items(memories)
    if not items:
        return "No memories found."

    lines = []
    for m in items[:limit]:
        if not isinstance(m, dict):
            lines.append(str(m))
            continue

        memory_id = m.get("id") or m.get("_id") or "?"
        text = (
            m.get("text")
            or m.get("memory")
            or m.get("data", {}).get("memory")
            or str(m)
        )
        meta = m.get("metadata") or m.get("meta") or {}
        source = meta.get("source") or ""
        important = meta.get("important")
        flags = []
        if source:
            flags.append(source)
        if important:
            flags.append("important")
        prefix = f"[{memory_id}]"
        if flags:
            prefix += " (" + ", ".join(flags) + ")"
        lines.append(f"{prefix} {text}")
    return "\n".join(lines)


# ---------------- Importance / storage heuristics ----------------

def should_store_memory(user_input: str, answer: str) -> bool:
    """
    Super simple heuristic for whether to store a turn as 'important'.
    This is intentionally cheap and local. You can later swap this out for
    an LLM-based classifier if you want.
    """
    text = (user_input + " " + answer).lower()
    keywords = [
        "remember",
        "note this",
        "important",
        "permanent",
        "preference",
        "goal",
        "target",
        "schedule",
        "budget",
    ]
    if any(k in text for k in keywords):
        return True
    # Also mark long / detailed inputs as important
    if len(user_input) > 140:
        return True
    return False


# ---------------- Core chat loop ----------------

def chat_once(user_input: str) -> str:
    global CURRENT_AGENT_ID

    # 1) search memories relevant to the input
    try:
        search_results = mem_client.search(query=user_input, agent_id=CURRENT_AGENT_ID) or []
    except Exception as e:
        print(f"[mem0 search error] {e}")
        search_results = []

    memory_block = format_memories(search_results, limit=8)

    # 2) build prompt with memory context
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "system",
            "content": f"Current agent: {CURRENT_AGENT_ID}",
        },
        {
            "role": "system",
            "content": "Relevant prior memories:\n" + memory_block,
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    # 3) call OpenAI
    resp = oa_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.4,
    )

    answer = resp.choices[0].message.content

    # 4) store this turn back into Mem0
    try:
        important = should_store_memory(user_input, answer)
        mem_client.add_memories(
            messages=[
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": answer},
            ],
            agent_id=CURRENT_AGENT_ID,
            metadata={"source": "local_assistant", "important": important},
        )
    except Exception as e:
        print(f"[mem0 add error] {e}")

    return answer


# ---------------- /mem commands ----------------

def handle_mem_command(cmd: str) -> str:
    """
    /mem                      -> list recent memories
    /mem search <query>       -> semantic search
    /mem show <id>            -> show a single memory
    /mem delete <id>          -> delete a single memory
    /mem clear                -> delete all memories for current agent
    """
    global CURRENT_AGENT_ID

    parts = cmd.strip().split(maxsplit=2)

    # just "/mem"
    if len(parts) == 1:
        try:
            res = mem_client.list_memories(agent_id=CURRENT_AGENT_ID)
        except Exception as e:
            return f"[mem0 list error] {e}"
        return format_memories(res, limit=30)

    sub = parts[1].lower() if len(parts) >= 2 else ""

    # "/mem search <query>"
    if sub == "search":
        if len(parts) == 2:
            return "Usage: /mem search <query>"
        query = parts[2]
        try:
            res = mem_client.search(query=query, agent_id=CURRENT_AGENT_ID)
        except Exception as e:
            return f"[mem0 search error] {e}"
        return format_memories(res, limit=30)

    # "/mem show <id>"
    if sub == "show":
        if len(parts) == 2:
            return "Usage: /mem show <memory_id>"
        memory_id = parts[2]
        try:
            m = mem_client.get_memory(memory_id=memory_id)
        except Exception as e:
            return f"[mem0 get error] {e}"
        return format_memories(m, limit=1)

    # "/mem delete <id>"
    if sub == "delete":
        if len(parts) == 2:
            return "Usage: /mem delete <memory_id>"
        memory_id = parts[2]
        try:
            res = mem_client.delete_memory(memory_id=memory_id)
        except Exception as e:
            return f"[mem0 delete error] {e}"
        return f"Deleted memory {memory_id}: {res}"

    # "/mem clear"
    if sub == "clear":
        try:
            res = mem_client.delete_all(agent_id=CURRENT_AGENT_ID)
        except Exception as e:
            return f"[mem0 clear error] {e}"
        return f"Cleared all memories for agent '{CURRENT_AGENT_ID}': {res}"

    return (
        "Unknown /mem command. Use:\n"
        "/mem\n"
        "/mem search <query>\n"
        "/mem show <id>\n"
        "/mem delete <id>\n"
        "/mem clear"
    )


# ---------------- /agent commands (multi-agent memory) ----------------

def handle_agent_command(cmd: str) -> str:
    """
    /agent                  -> show current agent
    /agent <name>           -> switch agent_id (separate memory space)
    """
    global CURRENT_AGENT_ID

    parts = cmd.strip().split(maxsplit=1)

    # just "/agent"
    if len(parts) == 1:
        return f"Current agent: {CURRENT_AGENT_ID}"

    new_agent = parts[1].strip()
    if not new_agent:
        return "Usage: /agent <name>"

    CURRENT_AGENT_ID = new_agent
    return f"Switched agent to: {CURRENT_AGENT_ID}"


# ---------------- Main loop ----------------

def main():
    print("Local assistant wired to Mem0.")
    print("Type 'exit' to quit.")
    print("Commands:")
    print("  /mem, /mem search <q>, /mem show <id>, /mem delete <id>, /mem clear")
    print("  /agent, /agent <name>")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        # /mem commands
        if user_input.startswith("/mem"):
            out = handle_mem_command(user_input)
            print("\n[MEMORIES]")
            print(out)
            continue

        # /agent commands
        if user_input.startswith("/agent"):
            out = handle_agent_command(user_input)
            print("\n[AGENT]")
            print(out)
            continue

        # normal chat
        reply = chat_once(user_input)
        print("\nAssistant:")
        print(textwrap.fill(reply, width=100))


if __name__ == "__main__":
    main()