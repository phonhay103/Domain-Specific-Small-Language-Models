"""
Implementing an AI Agent with SmolAgents.

Companion script for chapter 13 of "Domain-Specific Small Language Models"
by Guglielmo Iozzia, Manning Publications, 2025.

Implements a basic AI Agent using Hugging Face's SmolAgents framework
and Small Language Models (SLMs). Requires hardware acceleration.

# Install the missing requirement before running:
# pip install smolagents
"""

import random
import re

import torch
from smolagents import CodeAgent, TransformersModel, tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
MAX_NEW_TOKENS = 1000
AGENT_MAX_STEPS = 3
USER_TASK = "Find a flight from Dublin to New York and then select a seat."

# Mock webpage — prevents the agent from running longer queries across the web
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

6. Flight GWY-PHL303
- Price: $400
- Origin: Galway Airport (GWY)
- Destination: Philadelphia International Airport (PHL)
- Date: January 12, 2025

7. Flight GWY-CDG404
- Price: $150
- Origin: Galway Airport (GWY)
- Destination: Charles de Gaulle International Airport (CDG)
- Date: January 10, 2025

8. Flight DUB-PHL505
- Price: $390
- Origin: Dublin Airport (DUB)
- Destination: Philadelphia International Airport (PHL)
- Date: January 10, 2025
"""

FLIGHT_PATTERN = re.compile(
    r"Flight\s([A-Z]*-[A-Z]*[0-9]*)"
    r"- Price:\s+\$(\d+)"
    r"- Origin:\s+(.*?)"
    r"- Destination:\s+(.*?)"
    r"- Date:\s+([a-zA-Z]+\s[0-9]+,\s[0-9]+)"
)

# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

@tool
def search_for_flights(origin: str, destination: str, date: str, web_page: str) -> str:
    """Searches for flights based on origin, destination, and date.

    Args:
        origin: The origin airport code.
        destination: The destination airport code.
        date: The travel date.
        web_page: The webpage content to search for flights.

    Returns:
        The flight code of the matching flight.
    """
    origin = origin.lower()
    destination = destination.lower()

    # Always use the local mock page — for learning purposes only
    web_page = MOCK_FLIGHTS_WEB_PAGE
    flights = web_page.strip().split('\n\n')
    matching_flights = []

    for flight in flights:
        flight = flight.replace('\n', '')
        match = FLIGHT_PATTERN.search(flight)
        if match:
            flight_code = match.group(1)
            price = match.group(2)
            flight_origin = match.group(3).lower()
            flight_destination = match.group(4).lower()
            flight_date = match.group(5)

            if origin in flight_origin and destination in flight_destination:
                matching_flights.append({
                    "Flight Code": flight_code,
                    "Price": price,
                    "Origin": flight_origin,
                    "Destination": flight_destination,
                    "Date": flight_date,
                })

    return matching_flights


@tool
def find_seat(flight: str, seat_class: str = 'economy') -> str:
    """Finds a seat within a given flight.

    Args:
        flight: The flight code.
        seat_class: The desired seat class ('economy' or 'business').

    Returns:
        The seat number.
    """
    if seat_class.lower() not in ['business', 'economy']:
        raise ValueError("ticket_type must be either 'business' or 'economy'")

    seat_type = random.choice(['A', 'B', 'C', 'D', 'E', 'F'])

    if seat_class.lower() == 'business':
        seat_number = random.randint(1, 3)
    else:
        seat_number = random.randint(4, 32)

    return f"{seat_type}{seat_number}"


@tool
def book_flight_ticket(flight: str, seat: str) -> str:
    """Books a flight ticket.

    Args:
        flight: The flight code.
        seat: The seat number.

    Returns:
        The ticket number.
    """
    ticket_number = random.randint(100000, 999999)
    return f"Ticket booked for flight {flight} and seat {seat}. Ticket number: {ticket_number}"


@tool
def fibonacci(n: int) -> int:
    """Calculates the n-th Fibonacci number.

    Args:
        n: The sequence starting number.

    Returns:
        The n-th number in the sequence.
    """
    if n < 0:
        raise ValueError("Negative arguments are not supported")
    elif n == 0:
        return 0
    elif n == 1:
        return 1

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_model() -> TransformersModel:
    """Download and configure the SmolLM2-1.7B-Instruct model."""
    return TransformersModel(
        MODEL_ID,
        device_map="auto",
        max_new_tokens=MAX_NEW_TOKENS,
        torch_dtype=torch.float16,
    )


def build_agent(model: TransformersModel) -> CodeAgent:
    """Set up a CodeAgent with the three custom flight tools."""
    custom_tools = [search_for_flights, find_seat, book_flight_ticket]
    return CodeAgent(tools=custom_tools, model=model,
                     verbosity_level=2, max_steps=AGENT_MAX_STEPS)


def main() -> None:
    """Build the agent and run it on the user task."""
    # Quick self-tests for the tools (optional — remove if not needed)
    matching_flight = search_for_flights('DUB', 'JFK', '2023-03-15', MOCK_FLIGHTS_WEB_PAGE)
    print(matching_flight)

    selected_seat = find_seat('SFO-ANC123', 'Economy')
    print(selected_seat)

    booking = book_flight_ticket('DUB-JFK789', 'A1')
    print(booking)

    n = 10
    print(f"The {n}-th Fibonacci number is: {fibonacci(n)}")

    model = build_model()
    agent = build_agent(model)
    agent.run(USER_TASK)


if __name__ == "__main__":
    main()
