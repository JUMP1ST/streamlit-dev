import os
import json
import time
import yaml
import boto3
import logging
import requests
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

import streamlit as st
from streamlit_ace import st_ace
from io import BytesIO
from datetime import datetime
from typing import Any, Optional, List, Dict

# =====================================================
# Logging
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quasarhpc")

# =====================================================
# Config Manager
# =====================================================
class Config:
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str = None) -> dict:
        default_config = {
            "cluster": {
                "ray_dashboard_url": os.getenv("RAY_DASHBOARD_URL", "http://127.0.0.1:8265"),
                "ray_serve_url": os.getenv("RAY_SERVE_URL", "http://127.0.0.1:8265"),
                "namespace": os.getenv("NAMESPACE", "default"),
                "timeout": int(os.getenv("CLUSTER_TIMEOUT", "30")),
            },
            "storage": {
                "s3_endpoint": os.getenv("S3_ENDPOINT", ""),
                "s3_access_key": os.getenv("S3_ACCESS_KEY", ""),
                "s3_secret_key": os.getenv("S3_SECRET_KEY", ""),
                "s3_region": os.getenv("S3_REGION", "us-east-1"),
                "s3_use_ssl": os.getenv("S3_USE_SSL", "true").lower() == "true",
                "s3_verify_ssl": os.getenv("S3_VERIFY_SSL", "true").lower() == "true",
                "default_bucket": os.getenv("DEFAULT_BUCKET", ""),
            },
            "auth": {
                "enabled": os.getenv("AUTH_ENABLED", "false").lower() == "true",
            },
            "resources": {
                "max_cpus": int(os.getenv("MAX_CPUS", "64")),
                "max_memory_gb": int(os.getenv("MAX_MEMORY_GB", "256")),
                "max_gpus": int(os.getenv("MAX_GPUS", "8")),
                "default_cpus": int(os.getenv("DEFAULT_CPUS", "4")),
                "default_memory_gb": int(os.getenv("DEFAULT_MEMORY_GB", "16")),
            },
            "ui": {
                "title": os.getenv("UI_TITLE", "QuasarHPC Portal"),
                "theme": os.getenv("UI_THEME", "node"),
                "refresh_interval": int(os.getenv("REFRESH_INTERVAL", "5")),
            },
        }

        cm_path = "/etc/quasarhpc/config.yaml"
        if os.path.exists(cm_path):
            try:
                with open(cm_path, "r") as f:
                    file_cfg = yaml.safe_load(f) or {}
                    self._merge(default_config, file_cfg)
            except Exception as e:
                logger.warning(f"Failed to load ConfigMap: {e}")
        return default_config

    def _merge(self, base: dict, override: dict) -> None:
        for k, v in (override or {}).items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                self._merge(base[k], v)
            else:
                base[k] = v

    def get(self, key_path: str, default: Any = None) -> Any:
        cur = self.config
        for k in key_path.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    def set(self, key_path: str, value: Any) -> None:
        cur = self.config
        keys = key_path.split(".")
        for k in keys[:-1]:
            if k not in cur:
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = value

    def save_to_file(self, filepath: str) -> None:
        try:
            with open(filepath, "w") as f:
                yaml.safe_dump(self.config, f)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

config = Config()

# =====================================================
# Streamlit Config
# =====================================================
st.set_page_config(
    page_title=config.get("ui.title"),
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# Theme
# =====================================================
def get_theme_css(theme="node") -> str:
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');
    .stApp { background:#0b0b0b; color:#e0e0e0; font-family:'IBM Plex Mono', monospace; }
    h1,h2,h3,h4,h5 { color:#e0e0e0; text-transform:uppercase; letter-spacing:.05em; }
    .stButton>button { background:transparent; color:#d83cff; border:1px solid #333; border-radius:3px; }
    .stButton>button:hover { background:#1b1b1b; border-color:#d83cff; }
    section[data-testid="stSidebar"] { background:#101010; border-right:1px solid #1f1f1f; }
    </style>
    """

st.markdown(get_theme_css(), unsafe_allow_html=True)

# =====================================================
# Cluster & Storage Clients
# =====================================================
class ClusterAPI:
    def __init__(self):
        self.dashboard_url = config.get("cluster.ray_dashboard_url")
        self.timeout = config.get("cluster.timeout", 30)

    def get_cluster_status(self) -> Optional[dict]:
        """Query Ray cluster summary (supports new and old Ray versions)."""
        try:
            for path in ["/api/cluster_statuses", "/api/cluster_status", "/api/cluster_resources"]:
                url = f"{self.dashboard_url}{path}"
                r = requests.get(url, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
            logger.warning("No valid cluster status endpoint responded.")
        except Exception as e:
            logger.warning(f"Cluster unreachable: {e}")
        return None

    def get_jobs(self) -> List[dict]:
        """Return list of jobs from Ray Job API."""
        try:
            r = requests.get(f"{self.dashboard_url}/api/jobs", timeout=self.timeout)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict):
                    if "jobs" in data:
                        return data["jobs"]
                    if "data" in data:
                        return data["data"]
                    return []
                elif isinstance(data, list):
                    return data
            else:
                logger.warning(f"Job query failed: {r.status_code} {r.text}")
        except Exception as e:
            logger.warning(f"Job query failed: {e}")
        return []

    def submit_job(self, job_config: dict) -> Optional[dict]:
        """Submit a job to Ray Job API."""
        try:
            r = requests.post(f"{self.dashboard_url}/api/jobs", json=job_config, timeout=self.timeout)
            if r.status_code in (200, 201):
                return r.json()
            else:
                logger.warning(f"Job submission failed: {r.status_code} {r.text}")
        except Exception as e:
            logger.warning(f"Submit job error: {e}")
        return None

class StorageClient:
    def __init__(self):
        self.endpoint = config.get("storage.s3_endpoint")
        self.access_key = config.get("storage.s3_access_key")
        self.secret_key = config.get("storage.s3_secret_key")
        self.region = config.get("storage.s3_region")
        self.client = None

    def get_client(self):
        if not self.client and self.endpoint and self.access_key:
            self.client = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region
            )
        return self.client

api = ClusterAPI()
storage = StorageClient()

# =====================================================
# Sidebar Navigation
# =====================================================
with st.sidebar:
    st.title(config.get("ui.title"))
    section = st.radio("Navigation", ["Cluster Overview", "Python Lab", "Configuration"])

# =====================================================
# Cluster Overview
# =====================================================
if section == "Cluster Overview":
    st.header("Cluster Overview")

    refresh_rate = config.get("ui.refresh_interval", 5)
    st.caption(f"Auto-refresh every {refresh_rate}s")
    st_autorefresh = st.empty()
    time.sleep(refresh_rate)

    status = api.get_cluster_status()
    if not status:
        st.warning("Cluster not reachable. Check Ray Dashboard URL and try again.")
        if st.button("Refresh"):
            st.rerun()
        st.stop()

    total = status.get("cluster", {}).get("totalResources", {})
    used = status.get("cluster", {}).get("usedResources", {})
    nodes = status.get("nodes", [])

    cpu_total = float(total.get("CPU", 0))
    cpu_used = float(used.get("CPU", 0))
    st.metric("CPU Usage", f"{cpu_used}/{cpu_total}")

    st.subheader("Nodes")
    node_data = []
    for n in nodes:
        node_data.append({
            "Node": n.get("nodeId", "")[:10],
            "State": n.get("state", "Unknown"),
            "CPU": n.get("resources", {}).get("CPU", 0),
            "Memory": n.get("resources", {}).get("memory", 0),
        })
    if node_data:
        st.dataframe(pd.DataFrame(node_data), use_container_width=True)
    else:
        st.info("No active nodes detected.")

# =====================================================
# Python Lab
# =====================================================
elif section == "Python Lab":
    st.header("Python Lab")

    code = st_ace(
        value="# Example Python job\nprint('Hello from Ray!')",
        language="python",
        theme="monokai",
        key="lab_code",
        height=300,
    )

    if st.button("Submit Job"):
        job = {
            "entrypoint": "python -c \"print('Hello from Ray UI job')\"",
            "runtime_env": {},
        }
        result = api.submit_job(job)
        if result:
            st.success(f"Job submitted: {result.get('job_id', 'N/A')}")
        else:
            st.error("Failed to submit job — check Ray dashboard logs.")

    st.subheader("Active Jobs")
    jobs = api.get_jobs()
    if jobs:
        df = pd.DataFrame(jobs)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No jobs found or API unavailable.")

# =====================================================
# Configuration
# =====================================================
elif section == "Configuration":
    st.header("Configuration")
    st.json(config.config)

st.markdown("---")
st.caption(f"{config.get('ui.title')} · v1.0.1 (Ray API patched)")

