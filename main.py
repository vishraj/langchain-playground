import os
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
        Tool that searches over the internet
        Args:
            query: the query to search for
        Returns:
            The result of the search
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

def main():
    print("Hello from langchain-playground!")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        thinking_level="low",
        temperature=1.0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        timeout=30,
        max_retries=1)

    tools = [search]
    agent = create_agent(model=llm, tools=tools)

    result = agent.invoke({"messages": HumanMessage(content="What is the weather in Tokyo")})
    print(result)

if __name__ == "__main__":
    main()
