from typing import TypedDict, List, Union, Annotated,Sequence
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage, BaseMessage
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import os
from dotenv import load_dotenv
from langgraph.graph.message import add_messages


# === LLM setup
load_dotenv(override=True)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a:int,b:int)->str:
    """This Addition Funtion of Add 2 integers"""
    return f"Add tool return {a+b}"

tools =[add]

llm =ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    max_tokens=200,
    temperature=0.1,
    max_retries=2,
)

llm_with_tools = llm.bind_tools(tools)

def model_call(state:AgentState)->AgentState:
    SystemPrompt =SystemMessage(
        content="You are a helpful Assistand Called Alpo.In end of your response add Explaniation why this answer."
    )
    response = llm_with_tools.invoke([SystemPrompt]+state["messages"])
    # state["messages"].append(AIMessage(content=response.content))
    # return state
    return {"messages":[response]}

def should_continue(state:AgentState):
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "continue"
    else :
        return "end"

graph = StateGraph(AgentState)
graph.add_node("our_model",model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tools",tool_node)

graph.set_entry_point("our_model")
graph.add_conditional_edges(
    "our_model",
    should_continue,
    {
        "continue":"tools",
        "end":END,
    }
)

graph.add_edge("tools","our_model")
app = graph.compile()

def print_stream(stream):
    print("Enter the Stream Function.")
    counter=0
    for s in stream:
        counter+=1
        # LangGraph 'values' mode returns a dictionary where keys are node names
        # We get the 'messages' list from the state update and take the last one [-1]
        # print('*'*10)
        # print("S is ",s)
        # for key, value in s.items():
        #     if "messages" in value:
        message = s["messages"][-1]
        
        # Check if the message is a simple tuple (e.g., ("user", "hello"))
        if isinstance(message, tuple):
            print(message)
        else:
            # LangChain messages (AIMessage, HumanMessage, etc.) 
            # have a pretty_print() method for readable console output
            message.pretty_print()
    print("##"*10)
    print()

# Usage
inputs = {"messages": [("user", "Add 34 + 21 + 7")]}
print_stream(app.stream(inputs, stream_mode="values"))