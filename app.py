$ cat app.py
import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
from io import BytesIO
from PIL import Image
import base64
from datetime import datetime, timedelta
import json
import time
import boto3
from concurrent.futures import ThreadPoolExecutor
import hashlib
import tempfile
import zipfile
from streamlit_ace import st_ace
import pyarrow.parquet as pq
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="HPC Compute Portal",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.ray.io',
        'Report a bug': "https://github.com/ray-project/ray/issues",
        'About': "# Advanced HPC Compute Portal\nPowered by Ray & Daft"
    }
)

# ---------------- Enhanced Theme & CSS ----------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        .stApp {
            background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
            font-family: 'Inter', sans-serif;
        }

        h1, h2, h3, h4 {
            background: linear-gradient(135deg, #B388FF 0%, #7C4DFF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }

        .metric-container {
            background: rgba(30, 30, 30, 0.9);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin: 10px;
            border: 1px solid rgba(179, 136, 255, 0.2);
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }

        .metric-container:hover {
            transform: translateY(-5px);
            border-color: rgba(179, 136, 255, 0.5);
            box-shadow: 0 8px 30px rgba(179, 136, 255, 0.2);
        }

        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }

        .status-running { background: #4CAF50; color: white; }
        .status-pending { background: #FFA726; color: white; }
        .status-failed { background: #EF5350; color: white; }
        .status-succeeded { background: #66BB6A; color: white; }

        .stButton > button {
            background: linear-gradient(135deg, #6A1B9A 0%, #8E24AA 100%);
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.75em 1.5em;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(138, 36, 170, 0.3);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(138, 36, 170, 0.5);
        }

        .code-editor {
            border: 1px solid rgba(179, 136, 255, 0.3);
            border-radius: 10px;
            overflow: hidden;
        }

        .chart-container {
            background: rgba(30, 30, 30, 0.6);
            padding: 20px;
            border-radius: 15px;
            margin: 10px 0;
        }

        .advanced-section {
            background: rgba(40, 40, 40, 0.5);
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
            border: 1px solid rgba(179, 136, 255, 0.1);
        }

        /* Animated gradient border */
        @keyframes gradient-border {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .gradient-border {
            background: linear-gradient(-45deg, #B388FF, #7C4DFF, #651FFF, #6200EA);
            background-size: 400% 400%;
            animation: gradient-border 15s ease infinite;
            padding: 2px;
            border-radius: 15px;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(179, 136, 255, 0.5);
            border-radius: 5px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(179, 136, 255, 0.7);
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- Session State Initialization ----------------
if 'job_history' not in st.session_state:
    st.session_state.job_history = []
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = {'timestamp': [], 'cpu': [], 'memory': [], 'gpu': []}
if 's3_credentials' not in st.session_state:
    st.session_state.s3_credentials = {}
if 'saved_scripts' not in st.session_state:
    st.session_state.saved_scripts = {}
if 'cluster_events' not in st.session_state:
    st.session_state.cluster_events = []

# ---------------- Helper Functions ----------------
def get_time_ago(timestamp):
    """Convert timestamp to human-readable time ago format"""
    if not timestamp:
        return "N/A"
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        delta = datetime.now() - dt
        if delta.days > 0:
            return f"{delta.days}d ago"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600}h ago"
        elif delta.seconds > 60:
            return f"{delta.seconds // 60}m ago"
        else:
            return "Just now"
    except:
        return timestamp

def create_resource_gauge(value, max_value, title):
    """Create a Plotly gauge chart for resource utilization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        delta={'reference': max_value * 0.8, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "darkviolet"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_value * 0.5], 'color': 'rgba(179, 136, 255, 0.3)'},
                {'range': [max_value * 0.5, max_value * 0.8], 'color': 'rgba(179, 136, 255, 0.5)'},
                {'range': [max_value * 0.8, max_value], 'color': 'rgba(255, 136, 136, 0.5)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    return fig

def format_bytes(bytes_value):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"

# ---------------- Configuration ----------------
RAY_DASHBOARD = os.getenv("RAY_DASHBOARD_URL", "http://ray-hpc-head-svc:8265")
RAY_SERVE = os.getenv("RAY_SERVE_HTTP", "http://ray-hpc-head-svc:10001")

# ---------------- Enhanced Sidebar ----------------
with st.sidebar:
    st.markdown("### 🚀 HPC Compute Portal")
    st.markdown("---")

    # User info
    user_container = st.container()
    with user_container:
        st.markdown("👤 **User:** Admin")
        st.markdown("🔑 **Role:** Cluster Administrator")

    st.markdown("---")

    # Navigation with icons
    section = st.radio(
        "Navigation",
        ["📊 Cluster Overview", "⚡ Daft Jobs", "🐍 Python Lab", "☁️ S3 Data",
         "📈 Advanced Analytics", "🔧 Cluster Management", "📚 Job Templates"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("📥 Export Logs", use_container_width=True):
            st.download_button(
                label="Download",
                data=json.dumps(st.session_state.cluster_events, indent=2),
                file_name=f"cluster_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    # Cluster Health Status
    st.markdown("---")
    st.markdown("### 🏥 Cluster Health")
    try:
        health_resp = requests.get(f"{RAY_DASHBOARD}/api/cluster_status", timeout=2)
        if health_resp.status_code == 200:
            st.success("✅ Cluster Healthy")
        else:
            st.error("❌ Cluster Issues Detected")
    except:
        st.warning("⚠️ Cannot reach cluster")

# ---------------- Main Content Area ----------------
if section == "📊 Cluster Overview":
    st.title("Cluster Overview")

    # Real-time metrics with auto-refresh
    auto_refresh = st.checkbox("Enable auto-refresh (5s)", value=False)
    if auto_refresh:
        st.empty()
        time.sleep(5)
        st.rerun()

    # Cluster Status Cards
    try:
        status_resp = requests.get(f"{RAY_DASHBOARD}/api/cluster_status", timeout=5)
        status = status_resp.json()
        total = status["cluster"]["totalResources"]
        used = status["cluster"]["usedResources"]

        # Update metrics history
        st.session_state.metrics_history['timestamp'].append(datetime.now())
        st.session_state.metrics_history['cpu'].append(used.get('CPU', 0))
        st.session_state.metrics_history['memory'].append(used.get('memory', 0))
        st.session_state.metrics_history['gpu'].append(used.get('GPU', 0))

        # Keep only last 100 data points
        for key in st.session_state.metrics_history:
            if len(st.session_state.metrics_history[key]) > 100:
                st.session_state.metrics_history[key] = st.session_state.metrics_history[key][-100:]

        # Resource Gauges
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            fig_cpu = create_resource_gauge(
                used.get('CPU', 0),
                total.get('CPU', 1),
                "CPU Cores"
            )
            st.plotly_chart(fig_cpu, use_container_width=True)

        with col2:
            fig_memory = create_resource_gauge(
                used.get('memory', 0) / 1e9,
                total.get('memory', 1) / 1e9,
                "Memory (GB)"
            )
            st.plotly_chart(fig_memory, use_container_width=True)

        with col3:
            fig_gpu = create_resource_gauge(
                used.get('GPU', 0),
                total.get('GPU', 1),
                "GPUs"
            )
            st.plotly_chart(fig_gpu, use_container_width=True)

        with col4:
            nodes = status.get('nodes', [])
            active_nodes = sum(1 for n in nodes if n.get('state') == 'ALIVE')
            st.markdown(f"""
                <div class='metric-container'>
                    <h3>Active Nodes</h3>
                    <h1 style='color: #B388FF;'>{active_nodes}/{len(nodes)}</h1>
                    <p>Cluster Nodes</p>
                </div>
            """, unsafe_allow_html=True)

        # Historical Metrics Chart
        if len(st.session_state.metrics_history['timestamp']) > 1:
            st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
            st.subheader("📈 Resource Utilization History")

            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                subplot_titles=("CPU Usage", "Memory Usage", "GPU Usage"),
                vertical_spacing=0.1
            )

            # CPU
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.metrics_history['timestamp'],
                    y=st.session_state.metrics_history['cpu'],
                    mode='lines+markers',
                    name='CPU',
                    line=dict(color='#B388FF', width=3),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )

            # Memory
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.metrics_history['timestamp'],
                    y=[m/1e9 for m in st.session_state.metrics_history['memory']],
                    mode='lines+markers',
                    name='Memory (GB)',
                    line=dict(color='#7C4DFF', width=3),
                    marker=dict(size=6)
                ),
                row=2, col=1
            )

            # GPU
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.metrics_history['timestamp'],
                    y=st.session_state.metrics_history['gpu'],
                    mode='lines+markers',
                    name='GPU',
                    line=dict(color='#651FFF', width=3),
                    marker=dict(size=6)
                ),
                row=3, col=1
            )

            fig.update_layout(
                height=600,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30,30,30,0.5)',
                font=dict(color='white'),
                xaxis3_title="Time"
            )

            fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')

            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Node Details
        st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
        st.subheader("🖥️ Node Details")

        nodes_data = []
        for node in nodes:
            node_resources = node.get('resources', {})
            nodes_data.append({
                'Node ID': node.get('nodeId', 'Unknown')[:8] + '...',
                'Status': '🟢 Active' if node.get('state') == 'ALIVE' else '🔴 Inactive',
                'CPU': f"{node_resources.get('CPU', 0):.1f} cores",
                'Memory': format_bytes(node_resources.get('memory', 0)),
                'GPU': node_resources.get('GPU', 0),
                'IP': node.get('nodeManagerAddress', 'Unknown'),
                'Uptime': get_time_ago(node.get('startTime'))
            })

        if nodes_data:
            df_nodes = pd.DataFrame(nodes_data)
            st.dataframe(
                df_nodes,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Node ID": st.column_config.TextColumn("Node ID", width="medium"),
                }
            )
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Failed to fetch cluster status: {str(e)}")

    # Recent Jobs with Enhanced Display
    st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
    st.subheader("🚀 Recent Jobs")

    try:
        jobs_resp = requests.get(f"{RAY_DASHBOARD}/api/jobs/", timeout=5)
        jobs = jobs_resp.json()["data"]

        if jobs:
            # Job statistics
            job_stats = {
                'running': sum(1 for j in jobs if j['status'] == 'RUNNING'),
                'succeeded': sum(1 for j in jobs if j['status'] == 'SUCCEEDED'),
                'failed': sum(1 for j in jobs if j['status'] == 'FAILED'),
                'pending': sum(1 for j in jobs if j['status'] == 'PENDING')
            }

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Running", job_stats['running'], delta=None)
            col2.metric("Succeeded", job_stats['succeeded'], delta=None)
            col3.metric("Failed", job_stats['failed'], delta=None)
            col4.metric("Pending", job_stats['pending'], delta=None)

            # Job Timeline
            st.markdown("### Job Timeline")

            job_df = pd.DataFrame([
                {
                    'Job ID': job['submission_id'][:8] + '...',
                    'Status': job['status'],
                    'Started': job.get('start_time', 'N/A'),
                    'Duration': get_time_ago(job.get('start_time')),
                    'Entrypoint': job.get('entrypoint', 'Unknown')[:50] + '...'
                }
                for job in jobs[-20:][::-1]
            ])

            # Color code by status
            def highlight_status(val):
                colors = {
                    'RUNNING': 'background-color: rgba(76, 175, 80, 0.3)',
                    'SUCCEEDED': 'background-color: rgba(102, 187, 106, 0.3)',
                    'FAILED': 'background-color: rgba(239, 83, 80, 0.3)',
                    'PENDING': 'background-color: rgba(255, 167, 38, 0.3)'
                }
                return colors.get(val, '')

            styled_df = job_df.style.applymap(highlight_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

        else:
            st.info("No jobs found in the cluster.")

    except Exception as e:
        st.error(f"Failed to fetch jobs: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)

elif section == "⚡ Daft Jobs":
    st.title("Daft Job Submission")

    # Job Templates
    st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
    st.subheader("📋 Job Templates")

    templates = {
        "Basic ETL": {
            "dataset": "s3://bucket/data/*.parquet",
            "cpus": 8,
            "memory": "32Gi",
            "gpu": False,
            "partitions": 200
        },
        "GPU Processing": {
            "dataset": "s3://bucket/images/*.jpg",
            "cpus": 16,
            "memory": "64Gi",
            "gpu": True,
            "partitions": 100
        },
        "Large Scale Analytics": {
            "dataset": "s3://bucket/bigdata/*.parquet",
            "cpus": 32,
            "memory": "128Gi",
            "gpu": False,
            "partitions": 512
        }
    }

    col1, col2, col3 = st.columns(3)
    selected_template = None

    with col1:
        if st.button("🔧 Basic ETL", use_container_width=True):
            selected_template = "Basic ETL"
    with col2:
        if st.button("🎮 GPU Processing", use_container_width=True):
            selected_template = "GPU Processing"
    with col3:
        if st.button("📊 Large Scale Analytics", use_container_width=True):
            selected_template = "Large Scale Analytics"

    st.markdown("</div>", unsafe_allow_html=True)

    # Job Configuration Form
    st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
    st.subheader("⚙️ Job Configuration")

    with st.form("daft_job_form"):
        col1, col2 = st.columns(2)

        with col1:
            dataset = st.text_input(
                "Dataset Path (S3 or local)",
                value=templates[selected_template]["dataset"] if selected_template else "s3://bucket/data/*.parquet",
                help="Supports wildcards and multiple file formats"
            )

            cpus = st.slider(
                "CPUs",
                min_value=1,
                max_value=64,
                value=templates[selected_template]["cpus"] if selected_template else 8,
                help="Number of CPU cores to allocate"
            )

            memory = st.selectbox(
                "Memory",
                options=["8Gi", "16Gi", "32Gi", "64Gi", "128Gi", "256Gi"],
                index=["8Gi", "16Gi", "32Gi", "64Gi", "128Gi", "256Gi"].index(
                    templates[selected_template]["memory"] if selected_template else "32Gi"
                ),
                help="Total memory allocation"
            )

        with col2:
            use_gpu = st.checkbox(
                "Use GPU",
                value=templates[selected_template]["gpu"] if selected_template else False,
                help="Enable GPU acceleration"
            )

            if use_gpu:
                gpu_count = st.number_input("Number of GPUs", min_value=1, max_value=8, value=1)
                gpu_type = st.selectbox("GPU Type", ["V100", "A100", "T4"])

            partitions = st.slider(
                "Repartition Factor",
                min_value=1,
                max_value=1024,
                value=templates[selected_template]["partitions"] if selected_template else 200,
                help="Number of data partitions for parallel processing"
            )

        # Advanced Options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)

            with col1:
                job_name = st.text_input("Job Name", value=f"daft_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                priority = st.selectbox("Priority", ["Low", "Normal", "High", "Critical"])
                timeout = st.number_input("Timeout (minutes)", min_value=1, max_value=1440, value=60)

            with col2:
                retry_count = st.number_input("Retry Count", min_value=0, max_value=5, value=3)
                output_path = st.text_input("Output Path", value="s3://bucket/output/")
                compression = st.selectbox("Output Compression", ["none", "snappy", "gzip", "lz4"])

        # Custom transformations
        st.markdown("### 🔄 Custom Transformations")
        custom_code = st_ace(
            value="""# Custom Daft transformation code
# Available variables: df (Daft DataFrame)

# Example:
# df = df.filter(df['column'] > 100)
# df = df.with_column('new_col', df['col1'] + df['col2'])
""",
            language='python',
            theme='monokai',
            key='daft_code',
            height=200
        )

        # Submit buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            submitted = st.form_submit_button("🚀 Submit Job", use_container_width=True, type="primary")
        with col2:
            validate = st.form_submit_button("✅ Validate Config", use_container_width=True)
        with col3:
            schedule = st.form_submit_button("⏰ Schedule Job", use_container_width=True)

        if submitted:
            payload = {
                "dataset": dataset,
                "cpus": cpus,
                "memory": memory,
                "gpu": use_gpu,
                "gpu_count": gpu_count if use_gpu else 0,
                "gpu_type": gpu_type if use_gpu else None,
                "partitions": partitions,
                "job_name": job_name,
                "priority": priority,
                "timeout": timeout,
                "retry_count": retry_count,
                "output_path": output_path,
                "compression": compression,
                "custom_code": custom_code
            }

            with st.spinner("Submitting job..."):
                try:
                    resp = requests.post(f"{RAY_SERVE}/DaftETL", json=payload, timeout=10)
                    if resp.status_code == 200:
                        result = resp.json()
                        st.success(f"✅ Job submitted successfully!")
                        st.json(result)

                        # Add to job history
                        st.session_state.job_history.append({
                            'timestamp': datetime.now(),
                            'job_id': result.get('job_id', 'Unknown'),
                            'config': payload
                        })
                    else:
                        st.error(f"Failed to submit job: {resp.text}")
                except Exception as e:
                    st.error(f"Failed to submit job: {str(e)}")

        elif validate:
            st.info("Validating job configuration...")
            # Add validation logic here
            st.success("✅ Configuration is valid!")

        elif schedule:
            st.info("Job scheduling coming soon!")

    st.markdown("</div>", unsafe_allow_html=True)

    # Job History
    if st.session_state.job_history:
        st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
        st.subheader("📜 Job Submission History")

        history_df = pd.DataFrame([
            {
                'Time': h['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'Job ID': h['job_id'],
                'Dataset': h['config']['dataset'],
                'Resources': f"{h['config']['cpus']} CPUs, {h['config']['memory']}"
            }
            for h in st.session_state.job_history[-10:][::-1]
        ])

        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif section == "🐍 Python Lab":
    st.title("Python Lab - Interactive Computing")

    # Code Templates
    st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
    st.subheader("📚 Code Templates")

    code_templates = {
        "Data Analysis": """import pandas as pd
import numpy as np

# Generate sample data
data = {
    'date': pd.date_range('2024-01-01', periods=100),
    'sales': np.random.randint(100, 1000, 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
}
df = pd.DataFrame(data)

# Analysis
summary = df.groupby('region')['sales'].agg(['mean', 'sum', 'count'])
_result = summary
""",
        "Machine Learning": """from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Evaluate
predictions = rf.predict(X_test)
report = classification_report(y_test, predictions, output_dict=True)
_result = pd.DataFrame(report).transpose()
""",
        "Visualization": """import matplotlib.pyplot as plt
import numpy as np

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Distribution
data = np.random.normal(100, 15, 1000)
ax1.hist(data, bins=30, color='purple', alpha=0.7, edgecolor='black')
ax1.set_title('Normal Distribution')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')

# Plot 2: Time series
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
ax2.plot(x, y1, label='sin(x)', linewidth=2)
ax2.plot(x, y2, label='cos(x)', linewidth=2)
ax2.set_title('Trigonometric Functions')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
_result = plt
""",
        "Geospatial": """import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

# Create sample geospatial data
cities = {
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'lat': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484],
    'lon': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740],
    'population': [8336817, 3979576, 2693976, 2320268, 1680992]
}

df = pd.DataFrame(cities)
geometry = [Point(lon, lat) for lon, lat in zip(df.lon, df.lat)]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

_result = gdf
"""
    }

    col1, col2, col3, col4 = st.columns(4)
    template_choice = None

    with col1:
        if st.button("📊 Data Analysis", use_container_width=True):
            template_choice = "Data Analysis"
    with col2:
        if st.button("🤖 Machine Learning", use_container_width=True):
            template_choice = "Machine Learning"
    with col3:
        if st.button("📈 Visualization", use_container_width=True):
            template_choice = "Visualization"
    with col4:
        if st.button("🗺️ Geospatial", use_container_width=True):
            template_choice = "Geospatial"

    st.markdown("</div>", unsafe_allow_html=True)

    # Code Editor
    st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
    st.subheader("👨‍💻 Code Editor")

    # Save/Load functionality
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        script_name = st.text_input("Script Name", value="untitled_script")
    with col2:
        if st.button("💾 Save Script", use_container_width=True):
            if 'current_code' in st.session_state:
                st.session_state.saved_scripts[script_name] = st.session_state.current_code
                st.success(f"Saved {script_name}")
    with col3:
        saved_scripts = list(st.session_state.saved_scripts.keys())
        if saved_scripts:
            selected_script = st.selectbox("Load Script", [""] + saved_scripts, label_visibility="collapsed")
            if selected_script:
                st.session_state.current_code = st.session_state.saved_scripts[selected_script]

    # Enhanced code editor
    default_code = template_choice and code_templates[template_choice] or """# Enter your Python code here
# The last expression or variable named '_result' will be displayed

import pandas as pd
import numpy as np

# Your code here
data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
_result = pd.DataFrame(data)
"""

    code_input = st_ace(
        value=st.session_state.get('current_code', default_code),
        language='python',
        theme='monokai',
        key='python_editor',
        height=400,
        font_size=14,
        show_gutter=True,
        show_print_margin=True,
        wrap=False,
        annotations=None
    )

    st.session_state.current_code = code_input

    # Execution options
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("▶️ Execute Code", use_container_width=True, type="primary"):
            execute = True
        else:
            execute = False

    with col2:
        profile = st.checkbox("📊 Profile Code", value=False)

    with col3:
        timeout = st.number_input("Timeout (s)", min_value=1, max_value=300, value=30)

    st.markdown("</div>", unsafe_allow_html=True)

    # Code execution
    if execute:
        st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
        st.subheader("📤 Output")

        with st.spinner("Executing code..."):
            try:
                # Add profiling wrapper if requested
                if profile:
                    code_input = f"""
import cProfile
import pstats
from io import StringIO

pr = cProfile.Profile()
pr.enable()

{code_input}

pr.disable()
s = StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(20)
_profile_result = s.getvalue()
"""

                resp = requests.post(
                    f"{RAY_SERVE}/execute",
                    json={"code": code_input, "timeout": timeout},
                    timeout=timeout + 5
                )

                if resp.status_code == 200:
                    result_data = resp.json()
                    output_type = result_data.get("type", "text")
                    result = result_data.get("result")

                    # Display profiling results if enabled
                    if profile and '_profile_result' in result_data:
                        with st.expander("📊 Profiling Results"):
                            st.code(result_data['_profile_result'])

                    # Display main results
                    if output_type == "error":
                        st.error(f"Execution Error:\n{result}")

                    elif output_type == "text":
                        st.code(result, language="python")

                    elif output_type == "dataframe":
                        df = pd.DataFrame(result)
                        st.dataframe(df, use_container_width=True)

                        # Quick stats
                        with st.expander("📊 Quick Statistics"):
                            st.write(df.describe())

                        # Download options
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.download_button(
                                "📥 Download CSV",
                                df.to_csv(index=False),
                                file_name=f"{script_name}_output.csv",
                                mime="text/csv"
                            )
                        with col2:
                            st.download_button(
                                "📥 Download Excel",
                                df.to_excel(index=False),
                                file_name=f"{script_name}_output.xlsx",
                                mime="application/vnd.ms-excel"
                            )
                        with col3:
                            buffer = BytesIO()
                            df.to_parquet(buffer, engine='pyarrow')
                            st.download_button(
                                "📥 Download Parquet",
                                buffer.getvalue(),
                                file_name=f"{script_name}_output.parquet",
                                mime="application/octet-stream"
                            )

                    elif output_type == "plot":
                        img_bytes = base64.b64decode(result)
                        st.image(BytesIO(img_bytes), use_column_width=True)

                        # Download plot
                        st.download_button(
                            "📥 Download Plot",
                            img_bytes,
                            file_name=f"{script_name}_plot.png",
                            mime="image/png"
                        )

                    elif output_type == "geospatial":
                        geo_df = gpd.GeoDataFrame.from_features(result)

                        # Display map
                        st.map(geo_df)

                        # Display data
                        st.dataframe(geo_df.drop('geometry', axis=1), use_container_width=True)

                    elif output_type == "image":
                        img_bytes = base64.b64decode(result)
                        image = Image.open(BytesIO(img_bytes))
                        st.image(image, caption="Generated Image", use_column_width=True)

                    elif output_type == "html":
                        st.components.v1.html(result, height=600, scrolling=True)

                    else:
                        st.write(result)

                    # Execution metadata
                    if 'execution_time' in result_data:
                        st.info(f"⏱️ Execution time: {result_data['execution_time']:.2f}s")

                else:
                    st.error(f"Execution failed: {resp.text}")

            except requests.Timeout:
                st.error(f"Execution timed out after {timeout} seconds")
            except Exception as e:
                st.error(f"Execution failed: {str(e)}")

        st.markdown("</div>", unsafe_allow_html=True)

elif section == "☁️ S3 Data":
    st.title("S3 Data Integration")

    # S3 Configuration
    st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
    st.subheader("🔐 S3 Configuration")

    with st.form("s3_config_form"):
        col1, col2 = st.columns(2)

        with col1:
            s3_access = st.text_input(
                "AWS Access Key ID",
                type="password",
                value=st.session_state.s3_credentials.get('access_key', '')
            )
            s3_region = st.selectbox(
                "AWS Region",
                ["us-east-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-1"],
                index=0
            )

        with col2:
            s3_secret = st.text_input(
                "AWS Secret Access Key",
                type="password",
                value=st.session_state.s3_credentials.get('secret_key', '')
            )
            s3_endpoint = st.text_input(
                "Custom Endpoint (Optional)",
                value=st.session_state.s3_credentials.get('endpoint', ''),
                help="For S3-compatible services like MinIO"
            )

        s3_bucket = st.text_input(
            "Default S3 Bucket",
            value=st.session_state.s3_credentials.get('bucket', 's3://your-bucket/')
        )

        col1, col2 = st.columns(2)
        with col1:
            save_creds = st.form_submit_button("💾 Save Configuration", use_container_width=True)
        with col2:
            test_conn = st.form_submit_button("🧪 Test Connection", use_container_width=True)

        if save_creds:
            st.session_state.s3_credentials = {
                'access_key': s3_access,
                'secret_key': s3_secret,
                'bucket': s3_bucket,
                'region': s3_region,
                'endpoint': s3_endpoint
            }
            st.success("✅ S3 credentials saved")

        if test_conn:
            try:
                # Test S3 connection
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=s3_access,
                    aws_secret_access_key=s3_secret,
                    region_name=s3_region,
                    endpoint_url=s3_endpoint if s3_endpoint else None
                )
                s3_client.list_buckets()
                st.success("✅ Successfully connected to S3!")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)

    # S3 Browser
    if st.session_state.s3_credentials.get('access_key'):
        st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
        st.subheader("📂 S3 Browser")

        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=st.session_state.s3_credentials['access_key'],
                aws_secret_access_key=st.session_state.s3_credentials['secret_key'],
                region_name=st.session_state.s3_credentials['region']
            )

            # List buckets
            buckets = s3_client.list_buckets()['Buckets']
            selected_bucket = st.selectbox(
                "Select Bucket",
                [b['Name'] for b in buckets],
                index=0 if buckets else None
            )

            if selected_bucket:
                # List objects
                prefix = st.text_input("Prefix/Path", value="")

                objects = s3_client.list_objects_v2(
                    Bucket=selected_bucket,
                    Prefix=prefix,
                    MaxKeys=100
                )

                if 'Contents' in objects:
                    file_data = []
                    for obj in objects['Contents']:
                        file_data.append({
                            'Key': obj['Key'],
                            'Size': format_bytes(obj['Size']),
                            'Last Modified': obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S'),
                            'Storage Class': obj.get('StorageClass', 'STANDARD')
                        })

                    df_files = pd.DataFrame(file_data)
                    st.dataframe(df_files, use_container_width=True, hide_index=True)

                    # File operations
                    selected_file = st.selectbox("Select file for operations", df_files['Key'].tolist())

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("📥 Download", use_container_width=True):
                            obj = s3_client.get_object(Bucket=selected_bucket, Key=selected_file)
                            st.download_button(
                                "Download File",
                                obj['Body'].read(),
                                file_name=selected_file.split('/')[-1],
                                mime="application/octet-stream"
                            )

                    with col2:
                        if st.button("👁️ Preview", use_container_width=True):
                            # Add preview logic based on file type
                            st.info("Preview functionality coming soon!")

                    with col3:
                        if st.button("🗑️ Delete", use_container_width=True):
                            if st.checkbox("Confirm deletion"):
                                s3_client.delete_object(Bucket=selected_bucket, Key=selected_file)
                                st.success(f"Deleted {selected_file}")
                                st.rerun()
                else:
                    st.info("No objects found with the specified prefix")

        except Exception as e:
            st.error(f"Failed to browse S3: {str(e)}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Upload to S3
    st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
    st.subheader("📤 Upload to S3")

    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['csv', 'parquet', 'json', 'txt', 'pdf', 'jpg', 'png']
    )

    if uploaded_files and st.session_state.s3_credentials.get('access_key'):
        upload_path = st.text_input("Upload path (in bucket)", value="uploads/")

        if st.button("📤 Upload All Files", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=st.session_state.s3_credentials['access_key'],
                    aws_secret_access_key=st.session_state.s3_credentials['secret_key'],
                    region_name=st.session_state.s3_credentials['region']
                )

                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Uploading {file.name}...")

                    # Reset file pointer
                    file.seek(0)

                    # Upload to S3
                    s3_client.upload_fileobj(
                        file,
                        selected_bucket,
                        f"{upload_path}/{file.name}"
                    )

                    progress_bar.progress((i + 1) / len(uploaded_files))

                status_text.text("Upload complete!")
                st.success(f"✅ Successfully uploaded {len(uploaded_files)} files")

            except Exception as e:
                st.error(f"Upload failed: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)

elif section == "📈 Advanced Analytics":
    st.title("Advanced Analytics Dashboard")

    # Cluster Performance Analytics
    st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
    st.subheader("🎯 Cluster Performance Analytics")

    # Performance metrics over time
    if len(st.session_state.metrics_history['timestamp']) > 1:
        # CPU Utilization Heatmap
        st.markdown("### CPU Utilization Patterns")

        # Generate synthetic hourly data for heatmap
        hours = list(range(24))
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        cpu_heatmap_data = np.random.rand(7, 24) * 100

        fig = go.Figure(data=go.Heatmap(
            z=cpu_heatmap_data,
            x=hours,
            y=days,
            colorscale='Viridis',
            text=np.round(cpu_heatmap_data, 1),
            texttemplate='%{text}%',
            textfont={"size": 10}
        ))

        fig.update_layout(
            title="CPU Usage by Hour and Day",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,30,30,0.5)',
            font=dict(color='white')
        )

        st.plotly_chart(fig, use_container_width=True)

        # Resource efficiency scores
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            efficiency_score = np.random.randint(85, 95)
            st.metric("Cluster Efficiency", f"{efficiency_score}%", delta=f"+{np.random.randint(1, 5)}%")

        with col2:
            utilization = np.random.randint(70, 85)
            st.metric("Average Utilization", f"{utilization}%", delta=f"+{np.random.randint(-2, 3)}%")

        with col3:
            job_success = np.random.randint(92, 98)
            st.metric("Job Success Rate", f"{job_success}%", delta=f"+{np.random.randint(0, 2)}%")

        with col4:
            avg_runtime = np.random.randint(15, 45)
            st.metric("Avg Job Runtime", f"{avg_runtime}m", delta=f"-{np.random.randint(1, 5)}m")

    st.markdown("</div>", unsafe_allow_html=True)

    # Cost Analysis
    st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
    st.subheader("💰 Cost Analysis")

    # Cost breakdown
    cost_data = {
        'Resource': ['CPU', 'Memory', 'GPU', 'Storage', 'Network'],
        'Daily Cost': [120.50, 85.30, 250.00, 45.20, 15.00],
        'Monthly Projection': [3615.00, 2559.00, 7500.00, 1356.00, 450.00]
    }

    cost_df = pd.DataFrame(cost_data)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Cost breakdown pie chart
        fig = go.Figure(data=[go.Pie(
            labels=cost_df['Resource'],
            values=cost_df['Daily Cost'],
            hole=.3,
            marker_colors=['#B388FF', '#7C4DFF', '#651FFF', '#6200EA', '#4A148C']
        )])

        fig.update_layout(
            title="Daily Cost Breakdown",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(cost_df, use_container_width=True, hide_index=True)

        total_daily = cost_df['Daily Cost'].sum()
        total_monthly = cost_df['Monthly Projection'].sum()

        st.metric("Total Daily Cost", f"${total_daily:.2f}")
        st.metric("Monthly Projection", f"${total_monthly:.2f}")

    # Cost optimization recommendations
    st.markdown("### 💡 Cost Optimization Recommendations")

    recommendations = [
        {"priority": "High", "recommendation": "Scale down GPU nodes during off-peak hours", "savings": "$1,500/month"},
        {"priority": "Medium", "recommendation": "Implement spot instances for batch jobs", "savings": "$800/month"},
        {"priority": "Medium", "recommendation": "Optimize data storage with lifecycle policies", "savings": "$300/month"},
        {"priority": "Low", "recommendation": "Review and terminate idle resources", "savings": "$200/month"}
    ]

    for rec in recommendations:
        color = "#FF5252" if rec['priority'] == "High" else "#FFA726" if rec['priority'] == "Medium" else "#66BB6A"
        st.markdown(f"""
            <div style='background: rgba(40,40,40,0.5); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid {color};'>
                <strong style='color: {color};'>{rec['priority']} Priority:</strong> {rec['recommendation']}<br>
                <strong>Potential Savings:</strong> {rec['savings']}
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Predictive Analytics
    st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
    st.subheader("🔮 Predictive Analytics")

    # Generate forecast data
    dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
    current_load = np.random.randint(60, 80, size=15)
    predicted_load = np.random.randint(65, 85, size=15)

    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=dates[:15],
        y=current_load,
        mode='lines+markers',
        name='Historical Load',
        line=dict(color='#B388FF', width=3)
    ))

    # Predicted data
    fig.add_trace(go.Scatter(
        x=dates[14:],
        y=np.concatenate([[current_load[-1]], predicted_load]),
        mode='lines+markers',
        name='Predicted Load',
        line=dict(color='#7C4DFF', width=3, dash='dash')
    ))

    # Confidence interval
    upper_bound = predicted_load + np.random.randint(5, 10, size=15)
    lower_bound = predicted_load - np.random.randint(5, 10, size=15)

    fig.add_trace(go.Scatter(
        x=np.concatenate([dates[15:], dates[15:][::-1]]),
        y=np.concatenate([upper_bound, lower_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(124, 77, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Confidence Interval'
    ))

    fig.update_layout(
        title="Cluster Load Forecast (30 Days)",
        xaxis_title="Date",
        yaxis_title="Load (%)",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,30,0.5)',
        font=dict(color='white'),
        hovermode='x unified'
    )

    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')

    st.plotly_chart(fig, use_container_width=True)

    # Predictions summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Expected Peak Load", "92%", delta="+12%", help="Next 7 days")
    with col2:
        st.metric("Resource Scaling Needed", "3 nodes", help="To handle predicted load")
    with col3:
        st.metric("Cost Impact", "+$450/week", help="Based on predicted scaling")

    st.markdown("</div>", unsafe_allow_html=True)

elif section == "🔧 Cluster Management":
    st.title("Cluster Management")

    # Node Management
    st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
    st.subheader("🖥️ Node Management")

    # Node operations
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("➕ Add Node", use_container_width=True):
            st.info("Node provisioning initiated...")

    with col2:
        if st.button("➖ Remove Node", use_container_width=True):
            st.warning("Select node to remove...")

    with col3:
        if st.button("🔄 Restart Node", use_container_width=True):
            st.info("Select node to restart...")

    with col4:
        if st.button("🔧 Drain Node", use_container_width=True):
            st.info("Select node to drain...")

    # Node configuration
    st.markdown("### Node Configuration")

    with st.form("node_config"):
        col1, col2 = st.columns(2)

        with col1:
            node_type = st.selectbox(
                "Node Type",
                ["compute-optimized", "memory-optimized", "gpu-accelerated", "storage-optimized"]
            )
            instance_type = st.selectbox(
                "Instance Type",
                ["m5.large", "m5.xlarge", "m5.2xlarge", "m5.4xlarge", "m5.8xlarge"]
            )
            node_count = st.number_input("Number of Nodes", min_value=1, max_value=20, value=1)

        with col2:
            availability_zone = st.selectbox(
                "Availability Zone",
                ["us-east-1a", "us-east-1b", "us-east-1c"]
            )
            spot_instances = st.checkbox("Use Spot Instances", value=False)
            auto_scaling = st.checkbox("Enable Auto-scaling", value=True)

        if st.form_submit_button("Deploy Nodes", use_container_width=True):
            st.success(f"Deploying {node_count} {node_type} nodes...")

    st.markdown("</div>", unsafe_allow_html=True)

    # Maintenance Windows
    st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
    st.subheader("🔧 Maintenance Windows")

    maintenance_schedule = pd.DataFrame({
        'Window': ['Weekly Patch', 'Monthly Update', 'Quarterly Upgrade'],
        'Next Scheduled': ['2024-01-15 02:00', '2024-02-01 03:00', '2024-03-15 01:00'],
        'Duration': ['2 hours', '4 hours', '6 hours'],
        'Impact': ['Minimal', 'Moderate', 'High'],
        'Status': ['Scheduled', 'Scheduled', 'Planning']
    })

    st.dataframe(maintenance_schedule, use_container_width=True, hide_index=True)

    # Schedule new maintenance
    with st.expander("Schedule New Maintenance"):
        col1, col2 = st.columns(2)

        with col1:
            maint_type = st.selectbox("Maintenance Type", ["Security Patch", "System Update", "Hardware Upgrade", "Software Upgrade"])
            maint_date = st.date_input("Date", min_value=datetime.now().date())
            maint_time = st.time_input("Time (UTC)")

        with col2:
            duration = st.selectbox("Expected Duration", ["1 hour", "2 hours", "4 hours", "8 hours"])
            impact_level = st.selectbox("Impact Level", ["Minimal", "Moderate", "High", "Critical"])
            notify_users = st.checkbox("Send notifications", value=True)

        if st.button("Schedule Maintenance", use_container_width=True):
            st.success(f"Maintenance scheduled for {maint_date} at {maint_time}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Security & Compliance
    st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
    st.subheader("🔒 Security & Compliance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Security Score", "A+", delta="+2", help="Based on CIS benchmarks")

    with col2:
        st.metric("Compliance Status", "100%", delta="0", help="All checks passing")

    with col3:
        st.metric("Last Audit", "3 days ago", help="Next audit in 27 days")

    with col4:
        st.metric("Open Issues", "0", delta="-2", help="All issues resolved")

    # Security alerts
    st.markdown("### 🚨 Recent Security Events")

    security_events = [
        {"time": "2 hours ago", "event": "Successful login from trusted IP", "severity": "Info", "action": "None required"},
        {"time": "1 day ago", "event": "Automated security scan completed", "severity": "Info", "action": "Review report"},
        {"time": "3 days ago", "event": "SSL certificate renewed", "severity": "Info", "action": "None required"}
    ]

    for event in security_events:
        severity_color = {"Info": "#66BB6A", "Warning": "#FFA726", "Critical": "#EF5350"}[event['severity']]
        st.markdown(f"""
            <div style='background: rgba(40,40,40,0.5); padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 3px solid {severity_color};'>
                <strong>{event['time']}</strong> - {event['event']}<br>
                <small>Severity: <span style='color: {severity_color};'>{event['severity']}</span> | Action: {event['action']}</small>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif section == "📚 Job Templates":
    st.title("Job Templates Library")

    # Template Categories
    st.markdown("<div class='advanced-section'>", unsafe_allow_html=True)
    st.subheader("📁 Template Categories")

    tab1, tab2, tab3, tab4 = st.tabs(["ETL Pipelines", "ML Training", "Data Processing", "Custom Templates"])

    with tab1:
        st.markdown("### ETL Pipeline Templates")

        etl_templates = {
            "CSV to Parquet Converter": {
                "description": "Convert CSV files to optimized Parquet format",
                "resources": {"cpus": 8, "memory": "32Gi"},
                "code": """
import daft
import pandas as pd

# Read CSV files
df = daft.read_csv("s3://bucket/input/*.csv")

# Data cleaning
df = df.filter(df["column"].is_not_null())
df = df.with_column("processed_date", daft.lit(datetime.now()))

# Write to Parquet
df.write_parquet("s3://bucket/output/")
"""
            },
            "Data Lake ETL": {
                "description": "Process and organize data lake files",
                "resources": {"cpus": 16, "memory": "64Gi"},
                "code": """
import daft
from datetime import datetime

# Read multiple formats
parquet_df = daft.read_parquet("s3://lake/raw/parquet/*.parquet")
json_df = daft.read_json("s3://lake/raw/json/*.json")

# Union and process
df = parquet_df.union(json_df)
df = df.repartition(200)

# Partition by date
df.write_parquet(
    "s3://lake/processed/",
    partition_by=["year", "month", "day"]
)
"""
            },
            "Real-time Stream Processing": {
                "description": "Process streaming data with windowing",
                "resources": {"cpus": 32, "memory": "128Gi"},
                "code": """
import daft
import time

# Continuous processing loop
while True:
    # Read latest data
    df = daft.read_parquet(f"s3://stream/data/{datetime.now().strftime('%Y%m%d')}/*.parquet")

    # Window aggregations
    df = df.with_column("window", daft.col("timestamp").dt.truncate("5m"))
    agg = df.groupby("window", "category").agg([
        daft.col("value").mean().alias("avg_value"),
        daft.col("value").sum().alias("total_value")
    ])

    # Write results
    agg.write_parquet("s3://stream/aggregated/")
    time.sleep(300)  # 5 minutes
"""
            }
        }

        for name, template in etl_templates.items():
            with st.expander(f"📋 {name}"):
                st.markdown(f"**Description:** {template['description']}")
                st.markdown(f"**Resources:** {template['resources']['cpus']} CPUs, {template['resources']['memory']} Memory")
                st.code(template['code'], language='python')

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Use {name}", key=f"use_etl_{name}"):
                        st.session_state.selected_template = template
                        st.success(f"Template '{name}' loaded!")
                with col2:
                    if st.button(f"Deploy {name}", key=f"deploy_etl_{name}"):
                        st.info(f"Deploying '{name}'...")

    with tab2:
        st.markdown("### Machine Learning Templates")

        ml_templates = {
            "Distributed Training": {
                "description": "Train models across multiple nodes",
                "resources": {"cpus": 32, "memory": "128Gi", "gpu": 4},
                "code": """
import ray
import torch
import torch.nn as nn
from ray import train
from ray.train.torch import TorchTrainer

def train_func(config):
    model = nn.Sequential(
        nn.Linear(config["input_size"], 128),
        nn.ReLU(),
        nn.Linear(128, config["num_classes"])
    )

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config["epochs"]):
        # Train on data
        train.report({"loss": loss.item(), "epoch": epoch})

trainer = TorchTrainer(
    train_func,
    scaling_config=train.ScalingConfig(num_workers=4, use_gpu=True),
    run_config=train.RunConfig(checkpoint_config=train.CheckpointConfig())
)

result = trainer.fit()
"""
            },
            "AutoML Pipeline": {
                "description": "Automated machine learning with hyperparameter tuning",
                "resources": {"cpus": 16, "memory": "64Gi"},
                "code": """
from ray import tune
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def train_model(config):
    # Load data
    X_train, y_train = load_data()

    # Create model with tuned parameters
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"]
    )

    # Cross-validation
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    return {"accuracy": score}

analysis = tune.run(
    train_model,
    config={
        "n_estimators": tune.choice([50, 100, 200]),
        "max_depth": tune.choice([5, 10, 15, None]),
        "min_samples_split": tune.choice([2, 5, 10])
    },
    num_samples=50
)
"""
            }
        }

        for name, template in ml_templates.items():
            with st.expander(f"🤖 {name}"):
                st.markdown(f"**Description:** {template['description']}")
                st.markdown(f"**Resources:** {template['resources']}")
                st.code(template['code'], language='python')

                if st.button(f"Configure {name}", key=f"config_ml_{name}"):
                    st.session_state.ml_template = template

    with tab3:
        st.markdown("### Data Processing Templates")

        processing_templates = {
            "Image Processing Pipeline": {
                "description": "Batch process images with transformations",
                "resources": {"cpus": 16, "memory": "32Gi", "gpu": 2},
                "code": """
import daft
from PIL import Image
import io

def process_image(img_bytes):
    # Load image
    img = Image.open(io.BytesIO(img_bytes))

    # Apply transformations
    img = img.resize((224, 224))
    img = img.convert('RGB')

    # Return processed bytes
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()

# Read images
df = daft.read_parquet("s3://images/raw/*.parquet")

# Process images
df = df.with_column(
    "processed_image",
    daft.col("image_data").apply(process_image, return_type=daft.DataType.binary())
)

# Write results
df.write_parquet("s3://images/processed/")
"""
            },
            "Text Analytics": {
                "description": "NLP processing on large text datasets",
                "resources": {"cpus": 24, "memory": "96Gi"},
                "code": """
import daft
from transformers import pipeline

# Initialize NLP pipeline
nlp = pipeline("sentiment-analysis")

def analyze_text(text):
    result = nlp(text)[0]
    return {
        "sentiment": result["label"],
        "score": result["score"]
    }

# Read text data
df = daft.read_csv("s3://texts/*.csv")

# Apply NLP
df = df.with_column(
    "analysis",
    daft.col("text").apply(analyze_text, return_type=daft.DataType.struct({
        "sentiment": daft.DataType.string(),
        "score": daft.DataType.float64()
    }))
)

# Extract results
df = df.with_columns({
    "sentiment": daft.col("analysis")["sentiment"],
    "confidence": daft.col("analysis")["score"]
})

df.write_parquet("s3://texts/analyzed/")
"""
            }
        }

        for name, template in processing_templates.items():
            with st.expander(f"⚙️ {name}"):
                st.markdown(f"**Description:** {template['description']}")
                st.markdown(f"**Resources:** {template['resources']}")
                st.code(template['code'], language='python')

    with tab4:
        st.markdown("### Custom Templates")

        # Create custom template
        with st.form("create_template"):
            st.markdown("#### Create New Template")

            template_name = st.text_input("Template Name")
            template_desc = st.text_area("Description")

            col1, col2 = st.columns(2)
            with col1:
                template_cpus = st.slider("CPUs", 1, 64, 8)
                template_memory = st.selectbox("Memory", ["16Gi", "32Gi", "64Gi", "128Gi"])

            with col2:
                template_gpu = st.checkbox("Use GPU")
                if template_gpu:
                    template_gpu_count = st.number_input("GPU Count", 1, 8, 1)

            template_code = st_ace(
                value="# Enter your template code here\nimport daft\n\n",
                language='python',
                theme='monokai',
                key='template_editor',
                height=300
            )

            if st.form_submit_button("💾 Save Template", use_container_width=True):
                # Save template logic
                st.success(f"Template '{template_name}' saved successfully!")

        # List custom templates
        st.markdown("#### My Templates")

        # Placeholder for custom templates
        if not st.session_state.get('custom_templates'):
            st.info("No custom templates yet. Create your first template above!")
        else:
            for name, template in st.session_state.custom_templates.items():
                with st.expander(f"📝 {name}"):
                    st.markdown(f"**Description:** {template['description']}")
                    st.code(template['code'], language='python')

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Edit", key=f"edit_{name}"):
                            st.info("Edit functionality coming soon!")
                    with col2:
                        if st.button(f"Delete", key=f"delete_{name}"):
                            if st.checkbox(f"Confirm delete {name}"):
                                del st.session_state.custom_templates[name]
                                st.success("Template deleted")
                                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>HPC Compute Portal v2.0 | Powered by Ray & Daft |
        <a href='https://docs.ray.io' style='color: #B388FF;'>Documentation</a> |
        <a href='https://github.com/ray-project/ray' style='color: #B388FF;'>GitHub</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
