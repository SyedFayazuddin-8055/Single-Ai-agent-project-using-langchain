import os
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent


# --------------------------------------------------
# Load API keys
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Single AI Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Single AI Agent")
st.write("Ask me about current information, news, weather, and more.")


# --------------------------------------------------
# Check API keys
# --------------------------------------------------

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Add it to your .env file.")
    st.stop()

if not TAVILY_API_KEY:
    st.error("TAVILY_API_KEY is missing. Add it to your .env file.")
    st.stop()

if not WEATHERSTACK_API_KEY:
    st.warning(
        "WEATHERSTACK_API_KEY is missing. "
        "Weather queries will not work."
    )


# --------------------------------------------------
# Tavily Search Tool
# --------------------------------------------------

search_tool = TavilySearch(
    max_results=3
)


# --------------------------------------------------
# Weather Tool
# --------------------------------------------------

@tool
def get_weather(city: str) -> str:
    """Fetch current weather information for a city."""

    api_key = WEATHERSTACK_API_KEY

    if not api_key:
        return "WEATHERSTACK_API_KEY is not set."

    url = (
        "http://api.weatherstack.com/current"
        f"?access_key={api_key}"
        f"&query={city}"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if "current" not in data:
            return f"Could not fetch weather data for {city}."

        return (
            f"City: {city}\n"
            f"Temperature: {data['current']['temperature']}°C\n"
            f"Weather: {data['current']['weather_descriptions'][0]}\n"
            f"Humidity: {data['current']['humidity']}%"
        )

    except Exception as e:
        return f"Weather API error: {e}"


# --------------------------------------------------
# Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY
)


# --------------------------------------------------
# Agent
# --------------------------------------------------

tools = [
    search_tool,
    get_weather
]

agent = create_agent(
    model=llm,
    tools=tools
)


# --------------------------------------------------
# Chat interface
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
user_input = st.chat_input(
    "Ask something..."
)


if user_input:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)


    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = agent.invoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                })

                # Get final AI message
                final_message = response["messages"][-1]

                answer = final_message.content

                st.markdown(answer)

                # Save response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                error_message = f"Error: {e}"

                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
