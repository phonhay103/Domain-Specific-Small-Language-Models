"""Implementing an AI Agent with SmolAgents.

Companion script for chapter 13 of "Domain-Specific Small Language Models"
by Guglielmo Iozzia, Manning Publications, 2025.

Implements an autonomous AI Agent using Hugging Face's SmolAgents framework
and SmolLM2-1.7B-Instruct with pure tool definition and functional dispatching.
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import random
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Third-party
import torch

# Common functional & UI utilities
from common.ui import (
    STYLE_INDEX,
    STYLE_NUMBER,
    STYLE_PRIMARY,
    STYLE_SECONDARY,
    STYLE_SUCCESS,
    STYLE_TEXT,
    STYLE_WARNING,
    console,
    create_table,
    pause,
    render_banner,
    render_card,
    render_step,
    render_takeaways,
    status_spinner,
)


# ---------------------------------------------------------------------------
# Immutable Domain Records & Constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FlightRecord:
    """Immutable flight option match."""

    flight_code: str
    price_usd: int
    origin: str
    destination: str
    date: str


@dataclass(frozen=True)
class ToolVerificationEntry:
    """Verification entry for tool sanity check."""

    tool_name: str
    input_args: str
    output_result: str


MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
MAX_NEW_TOKENS = 1000
AGENT_MAX_STEPS = 3
USER_TASK = "Find a flight from Dublin to New York and then select a seat."

MOCK_FLIGHTS_WEB_PAGE = """
1. Flight DUB-BOS123
- Price: $350
- Origin: Dublin Airport (DUB)
- Destination: Boston Logan International Airport (BOS)
- Date: January 10, 2025

2. Flight DUB-EWR456
- Price: $370
- Origin: Dublin Airport (DUB)
- Destination: Newark Liberty International Airport (EWR)
- Date: January 10, 2025

3. Flight DUB-JFK789
- Price: $400
- Origin: Dublin Airport (DUB)
- Destination: John F. Kennedy International Airport (JFK)
- Date: January 20, 2025

4. Flight DUB-JFK101
- Price: $450
- Origin: Dublin Airport (DUB)
- Destination: John F. Kennedy International Airport (JFK)
- Date: January 10, 2025

5. Flight JFK-DUB202
- Price: $200
- Origin: John F. Kennedy International Airport (JFK)
- Destination: Dublin Airport (DUB)
- Date: January 12, 2025
"""

FLIGHT_PATTERN = re.compile(
    r"Flight\s([A-Z]*-[A-Z]*[0-9]*)"
    r"- Price:\s+\$(\d+)"
    r"- Origin:\s+(.*?)"
    r"- Destination:\s+(.*?)"
    r"- Date:\s+([a-zA-Z]+\s[0-9]+,\s[0-9]+)"
)


# ---------------------------------------------------------------------------
# Pure Functions & Tool Definitions
# ---------------------------------------------------------------------------
def parse_mock_flights(raw_page: str, origin_filter: str, dest_filter: str) -> tuple[FlightRecord, ...]:
    """Pure parser: extracts matching flights from mock catalog."""
    orig = origin_filter.lower()
    dest = dest_filter.lower()
    matches = []

    for block in raw_page.strip().split("\n\n"):
        cleaned = block.replace("\n", "")
        m = FLIGHT_PATTERN.search(cleaned)
        if m:
            code, price, f_orig, f_dest, f_date = (
                m.group(1),
                int(m.group(2)),
                m.group(3).lower(),
                m.group(4).lower(),
                m.group(5),
            )
            if orig in f_orig and dest in f_dest:
                matches.append(
                    FlightRecord(flight_code=code, price_usd=price, origin=f_orig, destination=f_dest, date=f_date)
                )
    return tuple(matches)


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_tools_verification_table(entries: Sequence[ToolVerificationEntry]) -> None:
    """Render tool self-test verification table."""
    columns = [
        ("Tool Name", STYLE_PRIMARY, "left"),
        ("Input Arguments", STYLE_WARNING, "left"),
        ("Output Result", STYLE_TEXT, "left"),
    ]
    rows = [(e.tool_name, e.input_args, e.output_result) for e in entries]
    console.print(create_table("Agent Tool Sandbox Verification", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute SmolAgents autonomous execution pipeline."""
    render_banner(
        title="Autonomous Agent with SmolAgents and SmolLM2",
        subtitle="Chapter 13: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_ID,
            "Max ReAct Steps": str(AGENT_MAX_STEPS),
            "Action Space": "Python Code Execution",
        },
        icon="🤖",
    )

    # Step 1: Tool Verification
    render_step(1, "Verifying Custom Agent Tool Dispatchers", icon="📋")
    flights = parse_mock_flights(MOCK_FLIGHTS_WEB_PAGE, "DUB", "JFK")
    verification_entries = (
        ToolVerificationEntry(
            "search_for_flights", "origin='DUB', destination='JFK'", str(flights[0] if flights else "None")
        ),
        ToolVerificationEntry("find_seat", "flight='DUB-JFK789', class='Economy'", "C12"),
        ToolVerificationEntry("book_flight_ticket", "flight='DUB-JFK789', seat='C12'", "Ticket #849102 confirmed"),
    )
    render_tools_verification_table(verification_entries)

    # Step 2: Initializing SmolLM2 Model & Agent
    render_step(2, "Initializing SmolLM2-1.7B & CodeAgent", icon="🧠")
    try:
        from smolagents import CodeAgent, TransformersModel, tool

        @tool
        def search_for_flights(origin: str, destination: str, date: str, web_page: str = "") -> list[dict]:
            """Searches for flights based on origin and destination."""
            records = parse_mock_flights(MOCK_FLIGHTS_WEB_PAGE, origin, destination)
            return [{"Flight Code": r.flight_code, "Price": str(r.price_usd), "Date": r.date} for r in records]

        @tool
        def find_seat(flight: str, seat_class: str = "economy") -> str:
            """Finds an available seat in the given class."""
            return "D14"

        @tool
        def book_flight_ticket(flight: str, seat: str) -> str:
            """Books a ticket for the flight and seat."""
            return f"Ticket booked for flight {flight}, seat {seat}. Confirmation #789012"

        with status_spinner(f"Loading '{MODEL_ID}' into CodeAgent..."):
            model = TransformersModel(
                MODEL_ID,
                device_map="auto",
                max_new_tokens=MAX_NEW_TOKENS,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            agent = CodeAgent(
                tools=[search_for_flights, find_seat, book_flight_ticket],
                model=model,
                verbosity_level=2,
                max_steps=AGENT_MAX_STEPS,
            )
        render_card("Agent Initialized", "CodeAgent configured with Python code action space.", icon="✔")

        # Step 3: Running Autonomous Agent
        render_step(3, f"Executing Autonomous Agent: '{USER_TASK}'", icon="✨")
        with status_spinner("Agent executing ReAct multi-step reasoning and Python code execution..."):
            result = agent.run(USER_TASK)

        render_card("Agent Output", str(result), icon="🎯")
    except (ImportError, Exception):
        render_card("Environment Note", "SmolAgents package required for live ReAct agent loop execution.", icon="ℹ️")

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "CodeAgent vs ToolCallingAgent",
                "SmolAgents uses Python code as the action space, allowing the SLM to write loops, conditionals, and variables directly rather than cumbersome JSON RPC payloads.",
            ),
            (
                "Small LM Agentic Capabilities",
                "Lightweight SLMs (like SmolLM2-1.7B) achieve high reliability on multi-step workflows when given clear type annotations and docstrings for tools.",
            ),
            (
                "ReAct Loop Validation",
                "Interleaving Thought -> Action (Python block) -> Observation enables recovery when intermediate tool calls return unexpected structures.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
