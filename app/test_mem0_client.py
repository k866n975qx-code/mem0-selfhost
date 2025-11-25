import os

from dotenv import load_dotenv
from app.mem0_client import Mem0Client

load_dotenv()

def main():
    client = Mem0Client(default_user_id="jose", default_agent_id="vera")

    # 1) Add a memory
    add_result = client.add_memories(
        messages=[
            {"role": "user", "content": "My name is Jose and I like high-yield ETFs."},
            {"role": "assistant", "content": "Noted your preference for high-yield dividend ETFs."},
        ]
    )
    print("ADD RESULT:", add_result)

    # 2) Search it back
    search_result = client.search(
        query="What does the user like to invest in?",
    )
    print("SEARCH RESULT:", search_result)


if __name__ == "__main__":
    main()