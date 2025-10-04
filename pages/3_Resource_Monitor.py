import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="Resource Monitor - HPC Console", layout="wide")

# Load custom CSS
with open('static/custom.css', 'r') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("RESOURCE MONITOR")

# Auto-refresh control
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    auto_refresh = st.checkbox("AUTO REFRESH", value=True)
with col2:
    refresh_rate = st.selectbox("INTERVAL", ["5s", "10s", "30s", "1m"])

if auto_refresh:
    time_value = int(refresh_rate.rstrip('sm'))
    time_unit = 1 if refresh_rate.endswith('s') else 60
    time.sleep(time_value * time_unit)
    st.rerun()

# Cluster overview
st.markdown("---")
st.subheader("CLUSTER OVERVIEW")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("TOTAL NODES", "24", delta="+2")
    st.caption("20 active, 4 idle")

with col2:
    st.metric("CPU CORES", "1,536", delta_color="off")
    st.caption("1,048 in use (68%)")

with col3:
    st.metric("TOTAL MEMORY", "6.0 TB", delta_color="off")
    st.caption("2.7 TB in use (45%)")

with col4:
    st.metric("GPU DEVICES", "32", delta_color="off")
    st.caption("28 in use (87.5%)")

# Real-time metrics
st.markdown("---")
st.subheader("REAL-TIME METRICS")

# Generate sample time-series data
time_points = pd.date_range(end=datetime.now(), periods=60, freq='1min')

# CPU Usage Chart
cpu_data = pd.DataFrame({
    'time': time_points,
    'node1': np.random.normal(70, 10, 60).clip(0, 100),
    'node2': np.random.normal(65, 8, 60).clip(0, 100),
    'node3': np.random.normal(80, 12, 60).clip(0, 100),
    'average': np.random.normal(72, 5, 60).clip(0, 100)
})

fig_cpu = go.Figure()
fig_cpu.add_trace(go.Scatter(
    x=cpu_data['time'], y=cpu_data['average'],
    mode='lines', name='AVERAGE',
    line=dict(color='#8a2be2', width=3)
))
fig_cpu.add_trace(go.Scatter(
    x=cpu_data['time'], y=cpu_data['node1'],
    mode='lines', name='NODE-1',
    line=dict(color='#6a0dad', width=1, dash='dot')
))
fig_cpu.update_layout(
    title="CPU USAGE ACROSS CLUSTER",
    paper_bgcolor='#0b0b0b',
    plot_bgcolor='#0b0b0b',
    font=dict(color='#e0e0e0'),
    height=300,
    showlegend=True,
    xaxis=dict(gridcolor='#2a2a2a'),
    yaxis=dict(gridcolor='#2a2a2a', range=[0, 100])
)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_cpu, use_container_width=True)

# Memory Usage Chart
memory_data = pd.DataFrame({
    'time': time_points,
    'used': np.random.normal(45, 3, 60).clip(0, 100),
    'cached': np.random.normal(20, 2, 60).clip(0, 100),
    'free': np.random.normal(35, 3, 60).clip(0, 100)
})

fig_mem = go.Figure()
fig_mem.add_trace(go.Scatter(
    x=memory_data['time'], y=memory_data['used'],
    mode='lines', fill='tozeroy', name='USED',
    line=dict(color='#8a2be2')
))
fig_mem.add_trace(go.Scatter(
    x=memory_data['time'], y=memory_data['cached'],
    mode='lines', fill='tonexty', name='CACHED',
    line=dict(color='#6a0dad')
))
fig_mem.update_layout(
    title="MEMORY USAGE",
    paper_bgcolor='#0b0b0b',
    plot_bgcolor='#0b0b0b',
    font=dict(color='#e0e0e0'),
    height=300,
    xaxis=dict(gridcolor='#2a2a2a'),
    yaxis=dict(gridcolor='#2a2a2a', range=[0, 100])
)

with col2:
    st.plotly_chart(fig_mem, use_container_width=True)

# Node status grid
st.markdown("---")
st.subheader("NODE STATUS")

nodes = []
for i in range(24):
    status = np.random.choice(['active', 'idle', 'maintenance'], p=[0.8, 0.15, 0.05])
    cpu = np.random.randint(0, 100) if status == 'active' else 0
    memory = np.random.randint(20, 80) if status == 'active' else 0
    temp = np.random.randint(50, 85) if status == 'active' else 45
    
    nodes.append({
        'name': f'NODE-{i+1:02d}',
        'status': status,
        'cpu': cpu,
        'memory': memory,
        'temp': temp
    })

# Display nodes in a grid
cols = st.columns(6)
for i, node in enumerate(nodes):
    with cols[i % 6]:
        status_color = {
            'active': '#00ff88',
            'idle': '#ffaa00',
            'maintenance': '#ff0044'
        }.get(node['status'], '#ffffff')
        
        st.markdown(f"""
            <div class="node-card" style="text-align: center; padding: 0.5rem;">
                <h4 style="margin: 0; font-size: 0.9rem;">{node['name']}</h4>
                <div style="width: 10px; height: 10px; background-color: {status_color}; 
                            border-radius: 50%; margin: 0.5rem auto;"></div>
                <div style="font-size: 0.7rem; color: var(--node-text-secondary);">
                    CPU: {node['cpu']}%<br>
                    MEM: {node['memory']}%<br>
                    TEMP: {node['temp']}°C
                </div>
            </div>
        """, unsafe_allow_html=True)

# GPU monitoring
st.markdown("---")
st.subheader("GPU MONITORING")

gpu_data = []
for i in range(8):
    gpu_data.append({
        'GPU': f'GPU-{i}',
        'Usage': np.random.randint(60, 95),
        'Memory': np.random.randint(40, 90),
        'Temperature': np.random.randint(65, 85),
        'Power': np.random.randint(150, 300)
    })

gpu_df = pd.DataFrame(gpu_data)

col1, col2 = st.columns([2, 3])

with col1:
    st.dataframe(
        gpu_df.style.background_gradient(subset=['Usage', 'Memory'], cmap='Purples'),
        use_container_width=True,
        height=300
    )

with col2:
    fig_gpu = go.Figure()
    fig_gpu.add_trace(go.Bar(
        x=gpu_df['GPU'],
        y=gpu_df['Usage'],
        name='Usage %',
        marker_color='#8a2be2'
    ))
    fig_gpu.add_trace(go.Bar(
        x=gpu_df['GPU'],
        y=gpu_df['Memory'],
        name='Memory %',
        marker_color='#6a0dad'
    ))
    fig_gpu.update_layout(
        title="GPU UTILIZATION",
        paper_bgcolor='#0b0b0b',
        plot_bgcolor='#0b0b0b',
        font=dict(color='#e0e0e0'),
        height=300,
        barmode='group'
    )
    st.plotly_chart(fig_gpu, use_container_width=True)

# Alerts
st.markdown("---")
st.subheader("SYSTEM ALERTS")

alerts = [
    {"level": "warning", "time": "2 min ago", "message": "Node-17 temperature above threshold (82°C)"},
    {"level": "error", "time": "5 min ago", "message": "GPU-5 memory allocation failed"},
    {"level": "warning", "time": "12 min ago", "message": "Network latency spike detected on subnet-3"},
    {"level": "info", "time": "18 min ago", "message": "Automatic backup completed successfully"},
]

for alert in alerts:
    level_color = {
        'error': 'var(--node-error)',
        'warning': 'var(--node-warning)',
        'info': 'var(--node-accent)'
    }.get(alert['level'], 'var(--node-text-secondary)')
    
    st.markdown(f"""
        <div class="node-card">
            <div style="display: flex; align-items: center;">
                <div style="width: 8px; height: 8px; background-color: {level_color}; 
                            border-radius: 50%; margin-right: 1rem;"></div>
                <div style="flex: 1;">
                    <span style="color: var(--node-text-secondary);">{alert['time']}</span> • 
                    {alert['message']}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
</content>
</invoke>
