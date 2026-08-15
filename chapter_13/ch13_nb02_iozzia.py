# ==========================================
# Extracted from CH13_NB02_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Implementing an AI Agent with SmolAgents
# This notebook is a companion of chapter 13 of the "Small Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2025.  
# The code in this notebook is about implementing a basic AI Agent using the Hugging Face's [SmolAgents](https://github.com/huggingface/smolagents) framework and Small Language Models (SLMs). Execution of the code cells in this notebook requires hardware acceleration.   
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the missing requirements (only SmolAgents missing in the Colab VM).
# ------------------------------------------------------------

# !pip install smolagents

# ------------------------------------------------------------
# Import the required packages and classes.
# ------------------------------------------------------------

import torch
from smolagents import CodeAgent, TransformersModel, tool

# ------------------------------------------------------------
# Because this notebook is for learning purposes only, let's define a mock webpage containing a list of flights, to prevent the agent running longer queries across the web.
# ------------------------------------------------------------

mock_flights_web_page = """
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

# ------------------------------------------------------------
# Let's implement the first custom tool for the agent. We achieve this by defining a function, to be decorated with the `@tool` decorator, that can extract from a web page a list of flights that match a given origin and destination pair on a given date. Please note that SmolAgents, for any tool implemented this way, requires type hints for inputs and outputs and docstrings that explain what the function does and what the arguments and return values are.
# ------------------------------------------------------------

import re

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

  web_page = mock_flights_web_page
  flights = web_page.strip().split('\n\n')
  matching_flights = []

  matching_pattern = r"Flight\s([A-Z]*-[A-Z]*[0-9]*)- Price:\s+\$(\d+)- Origin:\s+(.*?)- Destination:\s+(.*?)- Date:\s+([a-zA-Z]+\s[0-9]+,\s[0-9]+)"
  flight_pattern = re.compile(
      matching_pattern
      )

  for flight in flights:
      flight = flight.replace('\n', '')
      match = flight_pattern.search(flight)
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
                  "Date": flight_date
              })

  return matching_flights

# ------------------------------------------------------------
# The code cell below is to test the tool to search for flights. Execution isn't mandatory to run the agent.
# ------------------------------------------------------------

matching_flight = search_for_flights('DUB', 'JFK', '2023-03-15', mock_flights_web_page)
matching_flight

# ------------------------------------------------------------
# Define a function (tool) to find a seat within a given flight and class (Economy or Business).
# ------------------------------------------------------------

import random

@tool
def find_seat(flight: str, seat_class: str='economy') -> str:
  """Finds a seat within a given flight.

  Args:
    flight: The flight code.
    seat_class: The desired seat class.

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

  selected_seat = f"{seat_type}{seat_number}"

  return selected_seat

# ------------------------------------------------------------
# The code cell below is to test the tool to find a seat for a given flight. Execution isn't mandatory to run the agent.
# ------------------------------------------------------------

selected_seat = find_seat('SFO-ANC123', 'Economy')
selected_seat

# ------------------------------------------------------------
# Define a function (tool) to find book a ticket for a given flight and seat.
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# The code cell below is to test the tool to book a flight ticket. Execution isn't mandatory to run the agent.
# ------------------------------------------------------------

booking = book_flight_ticket('DUB-JFK789', 'A1')
booking

# ------------------------------------------------------------
# Define a function (tool) to calculate the n-th Fibonacci number.
# ------------------------------------------------------------

@tool
def fibonacci(n: int) -> int:
    """
    Calculates the n-th Fibonacci number.

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

# ------------------------------------------------------------
# The code cell below is to test the tool to calculate the n-th Fibonacci number. Execution isn't mandatory to run the agent.
# ------------------------------------------------------------

n = 10
print(f"The {n}-th Fibonacci number is: {fibonacci(n)}")

# ------------------------------------------------------------
# Download the instructed model ([HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)) to be used by the Agent.
# ------------------------------------------------------------

model = TransformersModel("HuggingFaceTB/SmolLM2-1.7B-Instruct",
                          device_map="auto",
                          max_new_tokens=1000,
                          torch_dtype=torch.float16)

# ------------------------------------------------------------
# Set up a CodeAgent, adding to its toolset the three custom tools for managing a flight search and booking and specifying the SLM to use.
# ------------------------------------------------------------

custom_tools = [search_for_flights, find_seat, book_flight_ticket]
agent = CodeAgent(tools=custom_tools, model=model,
                  verbosity_level=2, max_steps=3)

# ------------------------------------------------------------
# Run the agent on a user task and observe how it analyze it, splits it into steps, selects the tools, generates Python code and executes it.
# ------------------------------------------------------------

user_task = "Find a flight from Dublin to New York and then select a seat."
agent.run(user_task)


