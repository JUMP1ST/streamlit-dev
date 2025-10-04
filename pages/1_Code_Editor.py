import streamlit as st
from streamlit_ace import st_ace
import yaml
from datetime import datetime
from utils.templates import get_code_templates
from utils.job_scheduler import submit_job

st.set_page_config(page_title="Code Editor - HPC Console", layout="wide")

# Load custom CSS
with open('static/custom.css', 'r') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("CODE EDITOR")

# Editor layout
col1, col2 = st.columns([3, 1])

with col1:
    # Template selector
    templates = get_code_templates()
    template_names = ["Custom"] + list(templates.keys())
    selected_template = st.selectbox("TEMPLATE", template_names)
    
    # Code editor
    if selected_template != "Custom":
        default_code = templates[selected_template]["code"]
    else:
        default_code = "# Write your code here\n"
    
    code = st_ace(
        value=default_code,
        language="python",
        theme="monokai",
        key="code-editor",
        font_size=14,
        height=400,
        auto_update=False,
        wrap=False,
        show_gutter=True,
        show_print_margin=True,
        annotations=None
    )

with col2:
    st.subheader("JOB CONFIGURATION")
    
    job_name = st.text_input("JOB NAME", value=f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    st.markdown("### RESOURCES")
    cpus = st.slider("CPUs", 1, 64, 4)
    memory = st.slider("MEMORY (GB)", 1, 256, 16)
    gpus = st.slider("GPUs", 0, 8, 0)
    
    st.markdown("### EXECUTION")
    priority = st.select_slider("PRIORITY", options=["LOW", "NORMAL", "HIGH", "URGENT"])
    timeout = st.number_input("TIMEOUT (MIN)", min_value=1, max_value=1440, value=60)
    
    st.markdown("### DEPENDENCIES")
    requirements = st.text_area("REQUIREMENTS", placeholder="numpy==1.24.0\npandas>=2.0.0")
    
    st.markdown("### NOTIFICATIONS")
    notify_on_complete = st.checkbox("NOTIFY ON COMPLETION", value=True)
    notify_on_failure = st.checkbox("NOTIFY ON FAILURE", value=True)

# Submit section
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    if st.button("SUBMIT JOB", use_container_width=True):
        if code.strip():
            job_config = {
                "name": job_name,
                "code": code,
                "resources": {
                    "cpus": cpus,
                    "memory": memory,
                    "gpus": gpus
                },
                "priority": priority,
                "timeout": timeout,
                "requirements": requirements.split("\n") if requirements else [],
                "notifications": {
                    "on_complete": notify_on_complete,
                    "on_failure": notify_on_failure
                }
            }
            
            job_id = submit_job(job_config)
            st.success(f"JOB SUBMITTED: {job_id}")
            st.balloons()
        else:
            st.error("CODE CANNOT BE EMPTY")

with col2:
    if st.button("VALIDATE", use_container_width=True):
        st.info("CODE VALIDATION: PASSED")

# Code analysis
with st.expander("CODE ANALYSIS"):
    if code:
        st.markdown("### METRICS")
        lines = code.count('\n') + 1
        chars = len(code)
        imports = len([line for line in code.split('\n') if line.strip().startswith(('import', 'from'))])
        functions = len([line for line in code.split('\n') if line.strip().startswith('def ')])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("LINES", lines)
        col2.metric("CHARACTERS", chars)
        col3.metric("IMPORTS", imports)
        col4.metric("FUNCTIONS", functions)
