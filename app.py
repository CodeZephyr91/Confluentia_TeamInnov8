# streamlit_app.py

import streamlit as st
import base64
import io
import tempfile
import json
from agents import g, summary_gen, kpi_gen, generate_list_graph, generate_schema_dict, g_dashboard
from PIL import Image
import streamlit.components.v1 as components

st.set_page_config(page_title="Zingle DataBot", page_icon="🤖", layout="wide")

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Sidebar layout
st.sidebar.title("Zingle DataBot")
uploaded_db = st.sidebar.file_uploader("Upload SQLite DB", type=["sqlite"])
company_goals = st.sidebar.text_area("Company goals (for KPI suggestions)")

# If DB is uploaded
if uploaded_db:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
    tmp.write(uploaded_db.read())
    tmp.flush()
    conn_str = f"sqlite:///{tmp.name}"

    # Generate schema dynamically from uploaded DB
    schema = generate_schema_dict(conn_str)

    # Store in session state
    st.session_state["conn_str"] = conn_str
    st.session_state["schema"] = schema
    st.sidebar.success("Database uploaded and schema loaded!")

# Tabs: Chat, Schema, KPI
tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask Questions", "📚 Schema Summary", "📊 KPIs", "Dashboard"])

# -------------------- Chat --------------------
with tab1:
    st.title("Ask me anything about your database")

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if "graph" in m:
                st.image(m["graph"], caption=m["caption"])

    user_query = st.chat_input("Type your question about the data")
    if user_query:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    out = g.invoke({
                        "query": user_query,
                        "conn_str": st.session_state["conn_str"],
                        "schema": json.dumps(st.session_state["schema"])
                    })

                    sql = out["sql"]
                    table = out["table"]
                    reasoning=out["reasoning"]
                    graph_data = base64.b64decode(out["graph_data"])
                    caption, analysis = out["graph_caption"]

                    st.markdown(f"**Caption:** {caption}\n\n{analysis}")
                    st.image(graph_data)

                    with st.expander("View SQL query & data"):
                        st.code(sql, language="sql")
                        st.dataframe(table)
                    with st.expander("View Reasoning"):
                        st.write(reasoning)
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": f"**{caption}**\n\n{analysis}",
                        "graph": graph_data,
                        "caption": caption
                    })

                except Exception as e:
                    st.error(f"Error: {e}")

# -------------------- Schema Summary --------------------
with tab2:
    st.header("Schema Summary")
    if st.button("Generate summary"):
        if "schema" in st.session_state:
            with st.spinner("Analyzing schema..."):
                summary = summary_gen({
                    "schema": json.dumps(st.session_state["schema"])
                })
                st.write(summary["summary"])
        else:
            st.warning("Please upload a database first.")

# -------------------- KPIs --------------------
with tab3:
    st.header("KPIs")
    if st.button("Suggest KPIs"):
        if "schema" in st.session_state:
            with st.spinner("Generating KPIs..."):
                kpi = kpi_gen({
                    "schema": json.dumps(st.session_state["schema"]),
                    "goals": company_goals
                })
                st.write(kpi["kpi"])

            with st.spinner("Generating graphs..."):
                list_graph = generate_list_graph(
                    kpi["kpi"],
                    "For the given KPI, plot a graph highlighting the specific datapoints",
                    conn_str=st.session_state["conn_str"],
                    schema=st.session_state["schema"]
                )

            for item in list_graph:
                graph_bytes = base64.b64decode(item["graph_data"])
                st.image(graph_bytes, caption=item["graph_caption"][0])
                st.markdown(f"**Analysis:** {item['graph_caption'][1]}")
        else:
            st.warning("Please upload a database first.")

with tab4:
    st.header("Dashboard")
    if st.button("Suggest Dashboard"):
        if "schema" in st.session_state and "conn_str" in st.session_state:
            with st.spinner("Generating HTML dashboard..."):
                # Use session_state for schema and conn_str
                res = g_dashboard.invoke({
                    "schema": st.session_state["schema"],
                    "conn_str": st.session_state["conn_str"]  # pass from session_state
                })

                page_template = res["html_code"][0]
                graph_card_template = res["html_code"][1]

                cards_html = ""
                for graph_ in res["relevant_graphs"]:
                    cards_html += graph_card_template.format(
                        data=graph_["graph_data"],
                        caption=graph_["graph_caption"][0],
                        analysis=graph_["graph_caption"][1]
                    )

                final_html = page_template.format(cards=cards_html)
                components.html(final_html, height=900, scrolling=True)
        else:
            st.warning("Please upload a database first.")
