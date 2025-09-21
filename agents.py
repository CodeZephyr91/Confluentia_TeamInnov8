# agents.py

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import io
import base64
import json
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import OperationalError
from typing_extensions import TypedDict
import matplotlib.pyplot as plt
import seaborn

load_dotenv()

# -------------------- LLMs --------------------
llm = ChatGroq(model="openai/gpt-oss-120b",
               api_key=os.getenv("GROQ_API_KEY"))
llm_multimodal = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct",
                          api_key=os.getenv("GROQ_API_KEY"))

# -------------------- Helpers --------------------
def run_query(connection_string: str, query: str):
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        result = conn.execute(text(query))
        if result.returns_rows:
            return [dict(row._mapping) for row in result]
        else:
            return {"rows_affected": result.rowcount}

def generate_schema_dict(conn_str: str):
    """Generate schema dict dynamically from any SQLite DB connection string."""
    engine = create_engine(conn_str)
    inspector = inspect(engine)
    schema = {}
    for table in inspector.get_table_names():
        cols = inspector.get_columns(table)
        schema[table] = [
            {
                "name": c["name"],
                "type": str(c["type"]),
                "nullable": c["nullable"],
                "default": c["default"],
            }
            for c in cols
        ]
    return schema

# -------------------- Graph / SQL Agents --------------------
class GraphGenState(TypedDict):
    query: str
    conn_str: str
    schema: str
    sql: str
    table: dict
    graph_code: str
    graph_data: str
    graph_caption: tuple
    reasoning: str

def sql_gen(GraphGenState):
    messages = [SystemMessage("""
    You are an sql code generating agent. You are given a natural language query based on a SQL database.
    Your task is to write an SQL query that will answer the given natural language query. Only respond in a structured form with the SQL query only.
    IMPORTANT, generate only single sql query, multiple queries not allowed.
    No english text or 'Here's your code'. Just plain SQL. Not even stuff like "three backticks sql."
    ONLY RETURN SQL, no boilerplate stuff.
    """)]
    messages.append(HumanMessage(f"Here is the natural language query: {GraphGenState['query']}. Here is the database schema in JSON format: {GraphGenState['schema']}"))
    result = llm.invoke(messages)
    GraphGenState["sql"] = result.content
    GraphGenState["reasoning"]=result.additional_kwargs['reasoning_content']
    return GraphGenState

def sql_exec(GraphGenState):
    conn_str = GraphGenState["conn_str"]
    sql = GraphGenState["sql"]
    output = run_query(conn_str, sql)
    GraphGenState["table"] = output
    return GraphGenState

def graph_code_gen(GraphGenState):
    messages = [SystemMessage("""
    You are a matplotlib code generating agent. You are given a table. Think of a graph that can be plotted on the given table.
    Your task is to write python matplotlib code that can be executed out of the box.
    No english text or 'Here's your code'. Just plain python. Remember your output will be executed as it is, so generate responsibly.
    Import all the necessary modules you are using - very important.
    IMPORTANT, make sure to save the figure in a variable named fig. Make sure to add plt.close(fig) at the end.
    No english text or 'Here's your code'. Just plain python. Not even stuff like "three backticks python." DONT USE SEABORN- ONLY MATPLOTLIB.
    Dont use datetime- never.
    """)]
    messages.append(HumanMessage(f"Here is the query: {GraphGenState['query']}. Here is the table: {str(GraphGenState['table'])}"))
    result = llm.invoke(messages)
    GraphGenState["graph_code"] = result.content
    return GraphGenState

def graph_gen(GraphGenState):
   local_vars={}
   exec(GraphGenState["graph_code"], {}, local_vars)
   graph_fig = local_vars.get("fig") or local_vars.get("ax") or None
   buf = io.BytesIO()
   graph_fig.savefig(buf, format='png', bbox_inches='tight')
   GraphGenState["graph_data"] = base64.b64encode(buf.getvalue()).decode('utf-8')
   return GraphGenState

def caption_gen(GraphGenState):
    messages = [SystemMessage("""
        You are a caption and summary generating agent. You are given a graph and a schema for an SQL DB.
        Your task is to completely analyze the graph and
        1. Generate a caption for the graph, short and concise. not more than 10 words.
        2. Analyze the graph and provide insights on it in form of a short summary.
        IMPORTANT, return data in the form of a tuple like this ( <graph_caption> , <graph_analysis> ). Only return the tuple and nothing else. Make sure the tuple is properly formatted. Double check your output before sending it. If you are using inverted commas, properly escape them.
    """)]
    messages.append(HumanMessage(
        content = [
            {"type": "text", "text": f"Here is the schema: {str(GraphGenState['schema'])}"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{GraphGenState['graph_data']}"},  # fixed quotes
            }
        ]
    ))
    result = llm_multimodal.invoke(messages)
    GraphGenState["graph_caption"] = eval(result.content)
    return GraphGenState

# -------------------- Build main graph --------------------
builder = StateGraph(GraphGenState)
builder.add_node("SQL Generator", sql_gen)
builder.add_node("SQL Executor", sql_exec)
builder.add_node("Graph Code Generator", graph_code_gen)
builder.add_node("Graph Executor", graph_gen)
builder.add_node("Graph Caption", caption_gen)

builder.add_edge(START, "SQL Generator")
builder.add_edge("SQL Generator", "SQL Executor")
builder.add_edge("SQL Executor", "Graph Code Generator")
builder.add_edge("Graph Code Generator", "Graph Executor")
builder.add_edge("Graph Executor", "Graph Caption")
builder.add_edge("Graph Caption", END)

g = builder.compile()

# -------------------- Summary Agent --------------------
class SummaryGenState(TypedDict):
    schema: str
    summary: str

def summary_gen(state: SummaryGenState):
  messages = [SystemMessage("""
    You are a summary generating agent. You are given a schema for an sql db, you have to ananlyze all the tables in that schema and give a 2 line description for all the tables
    .Along with that, describe the relation between different tables taking into account their columns and titles and give a 1 line description of all the columns in every table
    as well in the format, column: description. The generation should be plain comprehensible text
  """)]
  messages.append(HumanMessage(f"Here is the schema: {str(state['schema'])}"))
  result = llm.invoke(messages)
  state["summary"] = result.content
  return state

builder_s = StateGraph(SummaryGenState)
builder_s.add_node("Summary Generator", summary_gen)
builder_s.add_edge(START, "Summary Generator")
builder_s.add_edge("Summary Generator", END)
g_s = builder_s.compile()

# -------------------- KPI Agent --------------------
class KPIGenState(TypedDict):
    schema: str
    goals: str
    kpi: list

def kpi_gen(state: KPIGenState):
  messages = [SystemMessage("""
  You are a KPI generating agent. You have been given the schema for the datatbase of a company along with the summary of the database and the goals of the company.
  Your task is to generate the 3 most relevant kpis based on these factors. Make sure these KPIs can be plotted on a graph. The data for these graphs will be fetched from the database, so think responsibly.
  IMPORTANT, return the kpis in the form of a list of strings. Only return the list and nothing else."
    """)]
  messages.append(HumanMessage(f"Here is the schema: {str(state['schema'])}"))
  messages.append(HumanMessage(f"Here are the goals: {str(state['goals'])}"))
  result=llm.invoke(messages)
  state["kpi"]=eval(result.content)
  return state

builder_k = StateGraph(KPIGenState)
builder_k.add_node("KPI Generator", kpi_gen)
builder_k.add_edge(START, "KPI Generator")
builder_k.add_edge("KPI Generator", END)
g_k = builder_k.compile()

# Dashboard Agent
class DashboardGenState(TypedDict):
    schema: str
    relevant_graph_names: list
    relevant_graphs: list
    html_code: tuple
    conn_str:str

def relevant_graph_name_gen(DashboardGenState):
    messages = [SystemMessage("""
    You are an advisor agent that advises what all graphs are relevant for a given database. You are given a SQL schema for a database with multiple tables that may or may not have relationships between them.
    Your goal is to generate ideas about graphs that may be relevant to show on a database dashboard. These ideas must be in english form. They will be later passed to an SQL query generator so double check what you propose can even exist or not.
    Be creative with your ideas, use different types of charts like line, bar, histogram, pie, boxplot, heatmaps. Whatever is relevant. Only generate 5 graphs.
    Output format: You have to output a python list of strings. Your output will be executed as it is in a python evaluator so make sure to return only a python list.
    """)]
    messages.append(HumanMessage(f"Here is the schema: {str(DashboardGenState['schema'])}"))
    result = llm.invoke(messages)
    DashboardGenState["relevant_graph_names"] = eval(result.content)
    return DashboardGenState
def relevant_graph_gen(DashboardGenState):
    graphs = []
    for graph_name in DashboardGenState["relevant_graph_names"]:
        try:
            graphs.append(g.invoke({
                "query": graph_name,
                "conn_str": DashboardGenState['conn_str'],
                "schema": str(DashboardGenState['schema'])
            }))
        except OperationalError as e:
            print("Schema mismatch error.")
    DashboardGenState["relevant_graphs"] = graphs
    return DashboardGenState

def dashboard_html_gen(DashboardGenState):
    messages = [SystemMessage("""
    You are an HTML coding agent. Your job is to design a clean, modern, and visually appealing dashboard layout in HTML. You are given the schema of the database and the names of the graphs that will be shown on the dashboard.
    Constraints & Requirements:
    Outputs: You must return a tuple of exactly two Python string literals. If you use inverted commas, make sure to escape them properly.
    - The first is the HTML page template, containing the full dashboard layout with inline CSS/JS, and a placeholder {cards} where multiple cards will be inserted.
    - The second is the graph card template, which defines the structure of a single card with placeholders {data}, {caption}, and {analysis}.
    Output format: 
    Both outputs must be returned as plain Python triple-quoted strings (\"\"\" ... \"\"\"). Do not use f-strings. Do not include explanations or comments outside of the strings.

    Embedding data:
    - {data} → base64 string for the graph image (used as <img src="data:image/png;base64,{data}">).
    - {caption} → short caption for the graph.
    - {analysis} → descriptive analysis text.
    - {cards} → concatenated string of card HTML blocks, inserted inside the page template.
    These placeholders will later be replaced by calling .format(...) in Python.

    Styling:
    - Include all CSS and JS inline in the page template. No external file references.
    - Layout should be responsive and professional, using card-like containers for each graph with image, caption, and analysis.

    Use a consistent theme: light background, rounded corners, padding, soft shadow.

    Scalability: The {cards} placeholder in the page template must support rendering any number of card blocks generated from the card template.

    Final Output:
    Only return a tuple of two Python triple-quoted strings: first the page template, second the card template. Nothing else.
    YOUR CODE WILL BE EXECUTED, SO PROPERLY ESCAPE EVERY SPECIAL CHARACTER, BRACES, INVERTED COMMAS.
    """)]
    graph_names = str(DashboardGenState["relevant_graph_names"])
    messages.append(HumanMessage(f"Here is the schema {DashboardGenState['schema']}. Here is the list of graph names: {graph_names}"))
    res = llm.invoke(messages)
    DashboardGenState["html_code"] = eval(res.content)
    return DashboardGenState

builder = StateGraph(DashboardGenState)
builder.add_node("Relevant Graph Name Generator", relevant_graph_name_gen)
builder.add_node("Relevant Graph Generator", relevant_graph_gen)
builder.add_node("Dashboard HTML Generator", dashboard_html_gen)

builder.add_edge(START, "Relevant Graph Name Generator")
builder.add_edge("Relevant Graph Name Generator", "Relevant Graph Generator")
builder.add_edge("Relevant Graph Generator", "Dashboard HTML Generator")
builder.add_edge("Dashboard HTML Generator", END)
g_dashboard = builder.compile()
g_dashboard.get_graph().print_ascii()

# -------------------- Multiple Graphs --------------------
def generate_list_graph(list_obj, query, conn_str, schema):
    list_graph = []
    for h in list_obj:
        try:
            list_graph.append(g.invoke({
                "query": query + f"{h}",
                "conn_str": conn_str,
                "schema": json.dumps(schema)
            }))
        except OperationalError as e:
            print("Schema mismatch error.", e)
    return list_graph

# -------------------- Exports --------------------
__all__ = [
    "g",
    "g_s",
    "g_k",
    "generate_list_graph",
    "run_query",
    "generate_schema_dict",
]