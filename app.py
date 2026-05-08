import streamlit as st
import pandas as pd
from databricks import sql
from groq import Groq

st.set_page_config(page_title="Data Validation Agent", layout="wide")
st.title("🔍 Data Validation Agent — powered by Groq AI")

# ── Read credentials from Streamlit secrets ────────────────────
host = st.secrets["https://dbc-2eebb4e7-9396.cloud.databricks.com"]
token = st.secrets["dapi4829d7a016e2a3427936c84149089af2"]
http_path = st.secrets["/sql/1.0/warehouses/1ce7ed1fb41a9370"]
groq_key = st.secrets["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"]

# ── Helper: Connect to Databricks ──────────────────────────────
def get_connection():
    return sql.connect(
        server_hostname=host.replace("https://", ""),
        http_path=http_path,
        access_token=token
    )

# ── Helper: Get all schemas and tables automatically ───────────
def get_all_tables():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SHOW SCHEMAS")
    schemas = [row[0] for row in cursor.fetchall()]
    all_tables = []
    for schema in schemas:
        try:
            cursor.execute(f"SHOW TABLES IN {schema}")
            for table in cursor.fetchall():
                all_tables.append(f"{schema}.{table[1]}")
        except Exception:
            pass
    cursor.close()
    conn.close()
    return all_tables

# ── Helper: Validate a single table ───────────────────────────
def validate_table(table_name):
    conn = get_connection()
    cursor = conn.cursor()

    # Row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cursor.fetchone()[0]

    # Column names + sample data
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
    columns = [desc[0] for desc in cursor.description]
    sample = cursor.fetchall()

    # Null count per column
    null_counts = {}
    for col in columns:
        cursor.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE `{col}` IS NULL"
        )
        null_counts[col] = cursor.fetchone()[0]

    # Duplicate rows
    cursor.execute(
        f"SELECT COUNT(*) - COUNT(DISTINCT *) FROM {table_name}"
    )
    duplicates = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    return {
        "table": table_name,
        "row_count": row_count,
        "columns": columns,
        "null_counts": null_counts,
        "duplicate_rows": duplicates,
        "sample": sample
    }

# ── Helper: Ask Groq AI to analyze results ─────────────────────
def analyze_with_groq(results):
    client = Groq(api_key=groq_key)
    prompt = f"""
You are a data quality expert. Analyze this validation report and explain:
1. What issues exist (nulls, duplicates, anomalies)
2. Which columns are most problematic
3. What action the data engineer should take

Report:
- Table: {results['table']}
- Total rows: {results['row_count']}
- Columns: {results['columns']}
- Null counts per column: {results['null_counts']}
- Duplicate rows: {results['duplicate_rows']}

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

# Auto-load tables on startup
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

# Table selector
st.subheader("📋 Select Tables to Validate")
selected = st.multiselect(
    "Choose one or more tables:",
    options=st.session_state["tables"],
    placeholder="Select tables here..."
)

# Run validation button
if st.button("🚀 Run Validation", use_container_width=True):
    if not selected:
        st.warning("⚠️ Please select at least one table first.")
    else:
        for table in selected:
            st.subheader(f"📊 Results: {table}")

            with st.spinner(f"Validating {table}..."):
                try:
                    results = validate_table(table)

                    # ── Metric cards ───────────────────────────
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Rows", f"{results['row_count']:,}")
                    c2.metric("Total Columns", len(results['columns']))
                    null_cols = sum(
                        1 for v in results['null_counts'].values() if v > 0
                    )
                    c3.metric("Columns with Nulls", null_cols)
                    c4.metric("Duplicate Rows", f"{results['duplicate_rows']:,}")

                    # ── Null counts bar chart ──────────────────
                    st.write("**Null counts by column:**")
                    null_df = pd.DataFrame(
                        list(results['null_counts'].items()),
                        columns=["Column", "Null Count"]
                    ).sort_values("Null Count", ascending=False)
                    st.bar_chart(null_df.set_index("Column"))

                    # ── Sample data ────────────────────────────
                    with st.expander("👀 View sample data (first 5 rows)"):
                        st.dataframe(
                            pd.DataFrame(
                                results['sample'],
                                columns=results['columns']
                            ),
                            use_container_width=True
                        )

                    # ── Groq AI Analysis ───────────────────────
                    st.write("**🤖 Groq AI Analysis:**")
                    with st.spinner("Asking Groq AI..."):
                        analysis = analyze_with_groq(results)
                    st.info(analysis)

                except Exception as e:
                    st.error(f"❌ Could not validate {table}: {e}")

            st.divider()
