"""

multi_agent.py

Streamlit app that implements a Multi-Agent Collaboration workflow using LangGraph,

translated from a Jupyter notebook into a Streamlit multi-agent page.

"""

import streamlit as st

import asyncio

import functools

import os

from typing import Sequence, Any, Literal

# LangChain / LangGraph imports (same as the notebook)

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.tools import tool, Tool

from langchain_experimental.utilities import PythonREPL

from ddgs import DDGS

from langchain.globals import set_debug

from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph

from langgraph.prebuilt import ToolNode

# Optional: matplotlib/pandas for any displayed output from python_repl (if returned)

import matplotlib.pyplot as plt

import pandas as pd

# ---------------------------

# Utilities / Validation

# ---------------------------

def validate_openai_key(key: str) -> bool:

    """Simple OpenAI key validation. Adjust as needed."""

    return bool(key and (key.startswith("sk-") or len(key) > 20))

# ---------------------------

# Streamlit page setup & UI helpers

# ---------------------------

st.set_page_config(page_title="Multi-Agent Collaboration", page_icon="", layout="centered")

def setup_header():

    st.title("Multi-Agent Collaboration")

    st.markdown("AI agents collaborating using LangGraph — create agents, tools, and a workflow graph.")

    st.caption("Data Science Dojo | Copyright (c) 2016-2025")

def show_info_rows():

    st.info(

        "This demo runs 2 agents (Researcher & Chart Generator) plus a Tool node. "

        "Agents can call tools and coordinate using a LangGraph workflow."

    )

# ---------------------------

# API key configuration (gating)

# ---------------------------

def configure_openai_key() -> bool:

    """

    Ensures OPENAI_API_KEY is available in session_state (or environment).

    Renders a small UI to enter it if missing.

    """

    openai_key = st.session_state.get("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))

    container = st.container()

    with container:

        if not openai_key:

            st.markdown("###  Enter OpenAI API Key")

            key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", key="openai_key_input")

            if st.button("Set Key", use_container_width=True):

                if validate_openai_key(key_input):

                    st.session_state["openai_api_key"] = key_input

                    os.environ["OPENAI_API_KEY"] = key_input

                    st.success(" OpenAI API key configured.")

                    st.experimental_rerun()

                else:

                    st.error(" Invalid OpenAI key format. It typically starts with `sk-`.")

            return False

        else:

            st.success(" OpenAI API key present.")

            # ensure environment variable is set for libraries that read it from env

            os.environ["OPENAI_API_KEY"] = openai_key

            return True

# ---------------------------

# Create agent helper (from notebook)

# ---------------------------

def create_agent(llm, tools, system_message: str):

    """

    Create an agent pipeline (ChatPromptTemplate | llm.bind_tools(tools))

    Mirroring notebook: system message + MessagesPlaceholder

    """

    prompt = ChatPromptTemplate.from_messages([

        (

            "system",

            "You are a helpful AI assistant, collaborating with other assistants."

            " Use the provided tools to progress towards answering the question."

            " If you are unable to fully answer, that's OK, another assistant with different tools "

            " will help where you left off. Execute what you can to make progress."

            " If you or any of the other assistants have the final answer or deliverable,"

            " prefix your response with FINAL ANSWER so the team knows to stop."

            " You have access to the following tools: {tool_names}.\n{system_message}",

        ),

        MessagesPlaceholder(variable_name="messages"),

    ])

    prompt = prompt.partial(system_message=system_message)

    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))

    return prompt | llm.bind_tools(tools)

# ---------------------------

# Define tools (from notebook)

# ---------------------------

# A search tool using DDGS and a python REPL tool

search = DDGS()

repl = PythonREPL()

@tool

def python_repl(code: str) -> str:

    """

    Execute python code via a REPL utility and return the output as text.

    This mirrors the notebook python_repl tool.

    """

    try:

        result = repl.run(code)

    except BaseException as e:

        return f"Failed to execute. Error: {repr(e)}"

    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

    # optionally instruct agent to finalize

    return result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."

# We'll expose a search tool as in the notebook

def search_text(query: str) -> Any:

    """Call DDGS search.text. Keep thin wrapper to satisfy Tool signature."""

    return search.text(query)

# Build a Tool object for search (langchain_core.tools.Tool)

search_tool_obj = Tool(

    name="Search",

    func=search_text,

    description="Useful for when you need to answer questions about current events",

)

# Also provide a @tool decorated search function (LangChain compatible)

@tool

def search_tool(text: str) -> str:

    """Search for information via DDGS."""

    try:

        result = search_text(text)

        return result

    except Exception as e:

        return f"Search failed: {e}"

# ---------------------------

# Agent node wrapper & agent creation

# ---------------------------

from langchain_core.messages import ToolMessage

from langgraph.graph import END, StateGraph

import operator  # used in notebook TypedDict typing (kept for parity)

def agent_node(state, agent, name: str):

    """

    Invocation wrapper used inside graph nodes.

    Accepts state, invokes the agent, and converts output into an AIMessage or ToolMessage.

    """

    result = agent.invoke(state)

    # If a ToolMessage: allow as-is; otherwise convert to AIMessage

    if isinstance(result, ToolMessage):

        # return tool invocation representation

        return {"messages": [result], "sender": name}

    else:

        # result likely a dataclass-like object with model_dump, as in notebook

        # Defensive: try to coerce to AIMessage

        try:

            ai_msg = AIMessage(**result.model_dump(exclude={"type", "name"}), name=name)

        except Exception:

            # fallback: create AIMessage with content str(result)

            ai_msg = AIMessage(content=str(result), name=name)

        return {"messages": [ai_msg], "sender": name}

# Create LLM

def build_llm():

    # Set debug similar to notebook for verbose logs if desired

    try:

        set_debug(True)

    except Exception:

        # set_debug may not be available or may behave differently — ignore errors

        pass

    # ChatOpenAI model selection - keep a stable model name; let environment control API key

    try:

        llm = ChatOpenAI(model="gpt-4o")

    except Exception:

        # fallback to default if model name unsupported in this environment

        llm = ChatOpenAI()

    return llm

# Create agents (researcher and chart generator)

def make_agents(llm):

    # Create search_tool and python_repl tool references for agent creation

    tools = [search_tool, python_repl]  # @tool decorated functions (LangChain will accept these)

    research_agent = create_agent(

        llm,

        [search_tool],  # researcher has search tool

        system_message="You should provide accurate data for the chart_generator to use.",

    )

    chart_agent = create_agent(

        llm,

        [python_repl],  # chart generator can call python repl to produce charts

        system_message="Any charts you display will be visible by the user. Make sure your code is safe and prints results.",

    )

    return research_agent, chart_agent, tools

# ---------------------------

# Graph construction (LangGraph)

# ---------------------------

def build_graph(research_agent, chart_agent, tools):

    # Define AgentState TypedDict equivalent structure the StateGraph will carry.

    # (langgraph.StateGraph expects a type/typing; we pass a simple dict-structure basis)

    workflow = StateGraph(dict)  # use dict so we don't have to recreate TypedDict

    # Node partials

    research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

    chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")

    # Tool node

    tool_node = ToolNode(tools)

    # Add nodes

    workflow.add_node("Researcher", research_node)

    workflow.add_node("chart_generator", chart_node)

    workflow.add_node("call_tool", tool_node)

    # Router (same logic as notebook)

    def router(state) -> Literal["call_tool", "__end__", "continue"]:

        messages = state["messages"]

        last_message = messages[-1]

        # If last message has tool_calls attribute (tool invocation), call tool

        if getattr(last_message, "tool_calls", None):

            return "call_tool"

        if "FINAL ANSWER" in getattr(last_message, "content", ""):

            return "__end__"

        return "continue"

    # Conditional edges

    workflow.add_conditional_edges(

        "Researcher",

        router,

        {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},

    )

    workflow.add_conditional_edges(

        "chart_generator",

        router,

        {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},

    )

    # After call_tool, we route back to the original sender stored in state["sender"]

    workflow.add_conditional_edges(

        "call_tool",

        lambda x: x["sender"],

        {"Researcher": "Researcher", "chart_generator": "chart_generator"},

    )

    workflow.set_entry_point("Researcher")

    graph = workflow.compile()

    return graph

# ---------------------------

# Stream/Invoke logic

# ---------------------------

def invoke_graph_and_stream(graph, human_prompt: str, max_steps: int = 150):

    """

    Run graph.stream similar to notebook and collect events. Returns a generator of events.

    This uses the synchronous generator returned by graph.stream; if asynchronous behavior

    is required we run it in an asyncio event loop in a blocking fashion similar to the agent example.

    """

    # Prepare initial payload similar to notebook: messages list with HumanMessage

    payload = {

        "messages": [HumanMessage(content=human_prompt)]

    }

    run_opts = {"recursion_limit": max_steps}

    # graph.stream returns an iterator/generator of events

    events_iter = graph.stream(payload, run_opts)

    return events_iter

# ---------------------------

# Streamlit app main

# ---------------------------

def main():

    setup_header()

    show_info_rows()

    # Ensure OPENAI key present

    if not configure_openai_key():

        return

    # initialize session state

    if "multi_messages" not in st.session_state:

        st.session_state.multi_messages = []  # list of {"role": "user"/"assistant"/"system", "content": str}

    if "multi_processing" not in st.session_state:

        st.session_state.multi_processing = False

    # Build LLM and agents (create once and cache)

    if "multi_llm_created" not in st.session_state:

        st.session_state.multi_llm = build_llm()

        # create agents and tools

        research_agent, chart_agent, tools = make_agents(st.session_state.multi_llm)

        st.session_state.multi_research_agent = research_agent

        st.session_state.multi_chart_agent = chart_agent

        st.session_state.multi_tools = tools

        # build graph

        st.session_state.multi_graph = build_graph(research_agent, chart_agent, tools)

        st.session_state.multi_llm_created = True

    # Display conversation history

    def render_messages():

        if not st.session_state.multi_messages:

            st.info("Ask a question to start the multi-agent collaboration. Example: 'Draw the chart that shows the population of world FROM 2023 TILL 2025.'")

        else:

            for msg in st.session_state.multi_messages:

                role = msg.get("role", "assistant")

                content = msg.get("content", "")

                if role == "user":

                    with st.chat_message("user"):

                        st.write(content)

                else:

                    with st.chat_message("assistant"):

                        st.write(content)

    render_messages()

    # Input area

    prompt_col, btn_col = st.columns([4,1])

    with prompt_col:

        user_prompt = st.chat_input("Enter your task for the agents...")

    with btn_col:

        pass

    # When user enters a prompt

    if user_prompt:

        st.session_state.multi_messages.append({"role": "user", "content": user_prompt})

        # Kick off graph execution

        # Prevent double submission

        if not st.session_state.multi_processing:

            st.session_state.multi_processing = True

            try:

                # Show a streaming assistant message container

                with st.chat_message("assistant"):

                    st.spinner_text = st.spinner("Agents collaborating and possibly calling tools...")

                    # Run graph.stream in a separate event loop (blocking until complete)

                    loop = asyncio.new_event_loop()

                    asyncio.set_event_loop(loop)

                    try:

                        events = loop.run_until_complete(

                            loop.run_in_executor(None, lambda: list(invoke_graph_and_stream(st.session_state.multi_graph, user_prompt)))

                        )

                    finally:

                        loop.close()

                    # events is a list of event objects (string representation in notebook)

                    # We'll append textual representation to messages. If events contain richer items,

                    # we attempt to extract .content or str()

                    for e in events:

                        try:

                            # Many graph event objects when printed output as strings in notebook;

                            # try to obtain a textual representation.

                            out_text = str(e)

                        except Exception:

                            out_text = repr(e)

                        st.session_state.multi_messages.append({"role": "assistant", "content": out_text})

                        # Stream each event to the UI as it is appended

                        st.write(out_text)

                        st.experimental_rerun()

                    st.success("Agents finished.")

            except Exception as ex:

                st.error(f"Graph execution error: {ex}")

            finally:

                st.session_state.multi_processing = False

    # Small section to show graph visualization if available

    st.markdown("---")

    st.subheader("Workflow Graph (rendered if available)")

    try:

        graph = st.session_state.multi_graph

        # attempt to get mermaid or PNG (not always available in Streamlit environment)

        img_data = None

        try:

            mermaid_png = graph.get_graph(xray=True).draw_mermaid_png()

            if mermaid_png:

                st.image(mermaid_png, caption="Workflow graph", use_column_width=True)

        except Exception as e:

            st.write("Graph rendering not available in this environment.", e)

    except Exception:

        st.write("Graph not initialized yet.")

    # Footer / help

    st.caption("This demo mirrors the Jupyter notebook workflow. Tools used: DDGS (DuckDuckGo), PythonREPL.")

if __name__ == "__main__":

    main()