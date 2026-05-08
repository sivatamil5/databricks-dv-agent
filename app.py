import streamlit as st
import pandas as pd
import requests
from groq import Groq

st.set_page_config(page_title="Data Validation Agent", layout="wide")
st.title("🔍 Data Validation Agent — powered by Groq AI")

host = st.secrets["DATABRICKS_HOST"]
token = st.secrets["DATABRICKS_TOKEN"]
groq_key = st.secrets["GROQ_API_KEY"]

headers = {"Authorization": f"Bearer {token}"}

# ── Get all catalogs, schemas, tables via REST API ─────────────
def get_all_tables():
    tables_list = []

    # Get catalogs
    resp = requests.get(
        f"{host}/api/2.1/unity-catalog/catalogs",
        headers=headers
    )
    if resp.status_code != 200:
        raise Exception(f"Cannot fetch catalogs: {resp.text}")

    catalogs = [c["name"] for c in resp.json().get("catalogs", [])]

    for catalog in catalogs:
        # Get schemas
        resp = requests.get(
            f"{host}/api/2.1/unity-catalog/schemas",
            headers=headers,
            params={"catalog_name": catalog}
        )
        if resp.status_code != 200:
            continue

        schemas = [s["name"] for s in resp.json().get("schemas", [])]

        for schema in schemas:
            # Get tables
            resp = requests.get(
                f"{host}/api/2.1/unity-catalog/tables",
                headers=headers,
                params={"catalog_name": catalog, "schema_name": schema}
            )
            if resp.status_code != 200:
                continue

            for t in resp.json().get("tables", []):
                tables_list.append({
                    "full_name": t["full_name"],
                    "table_type": t.get("table_type", ""),
                    "catalog": catalog,
                    "schema": schema,
                    "table": t["name"]
                })

    return tables_list

# ── Get table details via REST API ─────────────────────────────
def get_table_details(full_name):
    resp = requests.get(
        f"{host}/api/2.1/unity-catalog/tables/{full_name}",
        headers=headers
    )
    if resp.status_code != 200:
        raise Exception(f"Cannot fetch table details: {resp.text}")
    return resp.json()

# ── Ask Groq AI to analyze ─────────────────────────────────────
def analyze_with_groq(table_info):
    client = Groq(api_key=groq_key)
    prompt = f"""
You are a data quality expert. Analyze this table information and explain:
1. What the table contains based on column names
2. Any potential data quality concerns based on column types
3. What a data engineer should watch out for

Table info:
{table_info}

Keep it under 150 words. Be clear and beginner-friendly.
"""
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content

# ── Main app ───────────────────────────────────────────────────
st.markdown("---")

if "tables" not in st.session_state:
    with st.spinner("🔗 Connecting to Databricks and scanning tables..."):
        try:
            st.session_state["tables"] = get_all_tables()
            st.success(
                f"✅ Connected! Found {len(st.session_state['tables'])} tables."
            )
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")
            st.stop()

if "tables" in st.session_state:
    tables = st.session_state["tables"]

    if not tables:
        st.warning("⚠️ No tables found in your Databricks catalog.")
    else:
        st.subheader("📋 Select Tables to Validate")
        table_names = [t["full_name"] for t in tables]
        selected = st.multiselect(
            "Choose one or more tables:",
            options=table_names,
            placeholder="Select tables here..."
        )

        if st.button("🚀 Run Validation", use_container_width=True):
            if not selected:
                st.warning("⚠️ Please select at least one table first.")
            else:
                for table_name in selected:
                    st.subheader(f"📊 Results: {table_name}")
                    with st.spinner(f"Analyzing {table_name}..."):
                        try:
                            details = get_table_details(table_name)

                            columns = details.get("columns", [])
                            col_df = pd.DataFrame([
                                {
                                    "Column": c["name"],
                                    "Type": c.get("type_text", ""),
                                    "Nullable": c.get("nullable", True)
                                }
                                for c in columns
                            ])

                            c1, c2, c3 = st.columns(3)
                            c1.metric("Total Columns", len(columns))
                            nullable = sum(
                                1 for c in columns if c.get("nullable", True)
                            )
                            c2.metric("Nullable Columns", nullable)
                            c3.metric(
                                "Table Type",
                                details.get("table_type", "Unknown")
                            )

                            st.write("**Column details:**")
                            st.dataframe(col_df, use_container_width=True)

                            st.write("**🤖 Groq AI Analysis:**")
                            with st.spinner("Asking Groq AI..."):
                                analysis = analyze_with_groq(
                                    col_df.to_string()
                                )
                            st.info(analysis)

                        except Exception as e:
                            st.error(f"❌ Could not analyze {table_name}: {e}")

                    st.divider()
