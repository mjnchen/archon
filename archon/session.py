"""Session — thread agent state across multiple calls automatically.

Pass the same Session to repeated ``Agent.arun`` calls and the conversation
history persists. Equivalent to manually passing ``state=`` each time, but
keeps user code clean::

    session = Session()
    await agent.arun("Hi, my name is Alice.", session=session)
    await agent.arun("What's my name?", session=session)  # remembers Alice
"""

from __future__ import annotations

from typing import Optional

from archon.state import AgentState


class Session:
    """Holds the running ``AgentState`` between agent invocations."""

    def __init__(self) -> None:
        self.state: Optional[AgentState] = None

    def reset(self) -> None:
        """Discard accumulated history. Next call starts fresh."""
        self.state = None
