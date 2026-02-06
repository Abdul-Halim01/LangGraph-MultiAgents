from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import os
from dotenv import load_dotenv


# === LLM setup
load_dotenv(override=True)

# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     max_retries=2,
#     temperature=0.1,
#     max_tokens=500,
# )
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash-lite",
    temperature = 0.1,
    max_tokens = 500,
)

class AgentState(TypedDict):
    messages: List[Union[HumanMessage,AIMessage]]

def process(state: AgentState)->AgentState:
    response = llm.invoke(state["messages"])
    # return {"messages": state["messages"] + [response]}
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    print("CURRENT State: ",state['messages'])
    return state

conversation_history = []

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter your message: ")
counter=0
while user_input != "exit":
    counter+=1
    print(f"#{counter}")
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history=  result['messages']
    user_input = input("Enter your message: ")

with open("logging.txt", "w", encoding="utf-8") as file:
    file.write("Your Converstaion History!:\n")

    for message in conversation_history:
        if isinstance(message,HumanMessage):
            file.write(f"Human: {message.content}\n")
        elif isinstance(message,AIMessage):
            file.write(f"AI: {message.content}\n\n")    
        else:
            file.write(f"Unknown Message Type: {message}\n")
    file.write("End of Conversation!")

print("Converstaion Saved to logging.txt!")
