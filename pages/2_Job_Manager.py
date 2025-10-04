import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
from utils.job_scheduler import get_jobs, cancel_job, get_job_logs

st.set_page_config(page_title="Job Manager - HPC Console", layout="wide")

# Load custom CSS
with open('static/custom.css', 'r') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("JOB MANAGER")

# Filter controls
col1, col2, col3, col4 = st.columns(4)

with col1:
    status_filter = st.selectbox("STATUS", ["ALL", "RUNNING", "COMPLETED", "FAILED", "QUEUED"])
    
with col2:
    user_filter = st.text_input("USER", placeholder="Filter by user")
    
with col3:
    date_filter = st.date_input("DATE", value=datetime.now())
    
with col4:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("REFRESH", use_container_width=True):
        st.rerun()

# Job statistics
st.markdown("---")
st.subheader("JOB STATISTICS")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("TOTAL JOBS", "1,247", delta="+23")
with col2:
    st.metric("RUNNING", "47", delta="+5")
with col3:
    st.metric("QUEUED", "12", delta="-3")
with col4:
    st.metric("SUCCESS RATE", "94.3%", delta="+1.2%")
with col5:
    st.metric("AVG RUNTIME", "23.5 min", delta="-2.1 min")

# Job list
st.markdown("---")
st.subheader("ACTIVE JOBS")

# Mock job data
jobs = [
    {
        "id": "JOB-2024-1247",
        "name": "train_model_v3",
        "user": "alice",
        "status": "RUNNING",
        "progress": 65,
        "started": datetime.now() - timedelta(minutes=15),
        "runtime": "15m 32s",
        "cpus": "16/16",
        "memory": "32/64 GB",
        "gpus": "2/2"
    },
    {
        "id": "JOB-2024-1246",
        "name": "data_preprocessing",
        "user": "bob",
        "status": "RUNNING",
        "progress": 89,
        "started": datetime.now() - timedelta(minutes=45),
        "runtime": "45m 12s",
        "cpus": "8/8",
        "memory": "16/32 GB",
        "gpus": "0/0"
    },
    {
        "id": "JOB-2024-1245",
        "name": "feature_extraction",
        "user": "charlie",
        "status": "QUEUED",
        "progress": 0,
        "started": None,
        "runtime": "-",
        "cpus": "32",
        "memory": "128 GB",
        "gpus": "4"
    }
]

for job in jobs:
    with st.container():
        col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 2, 1])
        
        with col1:
            st.markdown(f"""
                <div>
                    <strong style="color: var(--node-accent);">{job['id']}</strong><br>
                    <span style="color: var(--node-text-secondary);">{job['name']}</span>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status_class = {
                "RUNNING": "warning",
                "COMPLETED": "success",
                "FAILED": "error",
                "QUEUED": ""
            }.get(job['status'], "")
            
            st.markdown(f"""
                <div style="margin-top: 0.5rem;">
                    <span class="node-status {status_class}">{job['status']}</span>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if job['status'] == "RUNNING":
                st.progress(job['progress'] / 100)
                st.caption(f"{job['progress']}% - Runtime: {job['runtime']}")
            elif job['status'] == "QUEUED":
                st.caption("Position in queue: 3")
        
        with col4:
            st.caption(f"Resources: {job['cpus']} CPUs, {job['memory']} RAM, {job['gpus']} GPUs")
            st.caption(f"User: {job['user']}")
        
        with col5:
            if st.button("DETAILS", key=f"details_{job['id']}"):
                st.session_state[f"show_details_{job['id']}"] = not st.session_state.get(f"show_details_{job['id']}", False)
            
            if job['status'] in ["RUNNING", "QUEUED"]:
                if st.button("CANCEL", key=f"cancel_{job['id']}"):
                    if st.checkbox(f"Confirm cancel {job['id']}?", key=f"confirm_{job['id']}"):
                        cancel_job(job['id'])
                        st.success(f"Job {job['id']} cancelled")
                        st.rerun()
        
        # Job details expansion
        if st.session_state.get(f"show_details_{job['id']}", False):
            with st.expander("JOB DETAILS", expanded=True):
                tab1, tab2, tab3, tab4 = st.tabs(["LOGS", "METRICS", "CONFIGURATION", "FILES"])
                
                with tab1:
                    st.code("""
[2024-03-15 14:32:01] INFO: Job started
[2024-03-15 14:32:02] INFO: Loading dataset...
[2024-03-15 14:32:15] INFO: Dataset loaded: 1.2M samples
[2024-03-15 14:32:16] INFO: Initializing model...
[2024-03-15 14:32:28] INFO: Training started
[2024-03-15 14:33:45] INFO: Epoch 1/100 - Loss: 0.6823, Accuracy: 0.7234
[2024-03-15 14:35:12] INFO: Epoch 2/100 - Loss: 0.5134, Accuracy: 0.8123
                    """, language="bash")
                
                with tab2:
                    # Metrics charts
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(10)),
                        y=[65, 68, 72, 71, 73, 75, 74, 76, 78, 77],
                        mode='lines',
                        name='GPU Usage %',
                        line=dict(color='#8a2be2')
                    ))
                    fig.update_layout(
                        paper_bgcolor='#0b0b0b',
                        plot_bgcolor='#0b0b0b',
                        font=dict(color='#e0e0e0'),
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    st.json({
                        "job_id": job['id'],
                        "resources": {
                            "cpus": 16,
                            "memory": "64GB",
                            "gpus": 2
                        },
                        "environment": {
                            "python": "3.11",
                            "cuda": "12.1"
                        },
                        "parameters": {
                            "batch_size": 128,
                            "learning_rate": 0.001,
                            "epochs": 100
                        }
                    })
                
                with tab4:
                    st.markdown("### INPUT FILES")
                    st.text("• /data/dataset_train.csv (2.3 GB)")
                    st.text("• /data/dataset_test.csv (512 MB)")
                    st.text("• /configs/model_config.yaml (2 KB)")
                    
                    st.markdown("### OUTPUT FILES")
                    st.text("• /outputs/model_checkpoint.pth (updating...)")
                    st.text("• /outputs/training_logs.txt (45 KB)")
                    st.text("• /outputs/metrics.json (12 KB)")
        
        st.markdown("<hr style='margin: 1rem 0; border-color: var(--node-border);'>", unsafe_allow_html=True)

# Batch operations
st.markdown("---")
st.subheader("BATCH OPERATIONS")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("CANCEL ALL QUEUED", use_container_width=True):
        st.warning("This will cancel 12 queued jobs. Proceed?")

with col2:
    if st.button("EXPORT HISTORY", use_container_width=True):
        st.info("Exporting job history to CSV...")

with col3:
    if st.button("CLEANUP COMPLETED", use_container_width=True):
        st.info("Removing completed jobs older than 7 days...")

with col4:
    if st.button("SYSTEM REPORT", use_container_width=True):
        st.info("Generating system utilization report...")
