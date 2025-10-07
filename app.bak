import streamlit as st
import yaml
from pathlib import Path
from utils.auth import check_authentication, get_user_role
import time

# Page configuration
st.set_page_config(
    page_title="HPC Console",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "HPC Compute Console v2.0"
    }
)

# Load custom CSS
with open('static/custom.css', 'r') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load configuration
with open('config/settings.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None

# Login form
if not st.session_state.authenticated:
    st.markdown('<h1 style="text-align: center;">HPC CONSOLE LOGIN</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("USERNAME", placeholder="Enter username")
            password = st.text_input("PASSWORD", type="password", placeholder="Enter password")
            submit = st.form_submit_button("AUTHENTICATE", use_container_width=True)
            
            if submit:
                if check_authentication(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.role = get_user_role(username)
                    st.rerun()
                else:
                    st.error("AUTHENTICATION FAILED")
else:
    # Main application
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
            <h1 style="margin: 0;">HPC COMPUTE CONSOLE</h1>
            <div style="text-align: right;">
                <span style="color: var(--node-text-secondary);">USER: {st.session_state.username.upper()}</span><br>
                <span style="color: var(--node-accent);">ROLE: {st.session_state.role.upper()}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # System status bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CLUSTER STATUS", "ONLINE", delta="100%")
    with col2:
        st.metric("ACTIVE JOBS", "47", delta="+5")
    with col3:
        st.metric("CPU USAGE", "68%", delta="-12%")
    with col4:
        st.metric("MEMORY USAGE", "45%", delta="+3%")
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("QUICK ACTIONS")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("NEW JOB", use_container_width=True):
            st.switch_page("pages/1_Code_Editor.py")
    
    with col2:
        if st.button("VIEW JOBS", use_container_width=True):
            st.switch_page("pages/2_Job_Manager.py")
    
    with col3:
        if st.button("MONITOR", use_container_width=True):
            st.switch_page("pages/3_Resource_Monitor.py")
    
    with col4:
        if st.button("STORAGE", use_container_width=True):
            st.switch_page("pages/6_Storage_Browser.py")
    
    # Recent activity
    st.markdown("---")
    st.subheader("RECENT ACTIVITY")
    
    activities = [
        {"time": "2 min ago", "user": "alice", "action": "Submitted job #1247", "status": "success"},
        {"time": "5 min ago", "user": "bob", "action": "Updated model registry", "status": "success"},
        {"time": "12 min ago", "user": "system", "action": "Auto-scaled cluster", "status": "warning"},
        {"time": "18 min ago", "user": "charlie", "action": "Data pipeline completed", "status": "success"},
        {"time": "25 min ago", "user": "alice", "action": "Job #1244 failed", "status": "error"},
    ]
    
    for activity in activities:
        status_class = activity['status']
        st.markdown(f"""
            <div class="node-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: var(--node-text-secondary);">{activity['time']}</span> • 
                        <span style="color: var(--node-accent);">{activity['user'].upper()}</span> • 
                        {activity['action']}
                    </div>
                    <span class="node-status {status_class}">{status_class.upper()}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Logout button
    if st.sidebar.button("LOGOUT"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.role = None
        st.rerun()
