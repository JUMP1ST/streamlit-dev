import os, sys, json, time, traceback
import ray

def _print(msg):  # simple timestamped logger
    print(f"[BOOTSTRAP] {time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}", flush=True)

def upload_results_if_any():
    s3_uri = os.getenv("RESULTS_S3_URI")
    if not s3_uri:
        return
    try:
        import boto3, urllib.parse, pathlib
        p = pathlib.Path("results.json")
        if p.exists():
            bucket = s3_uri.replace("s3://","").split("/",1)[0]
            key = s3_uri.replace(f"s3://{bucket}/", "")
            s3 = boto3.client("s3",
                              endpoint_url=os.getenv("AWS_ENDPOINT_URL",""),
                              aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                              aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                              region_name=os.getenv("AWS_DEFAULT_REGION","us-east-1"))
            s3.upload_file(str(p), bucket, key)
            _print(f"Uploaded results.json to {s3_uri}")
    except Exception as e:
        _print(f"WARNING: upload failed: {e}")

def main():
    ray_addr = os.getenv("RAY_ADDRESS", "auto")
    _print(f"Connecting to Ray at: {ray_addr}")
    ray.init(address=ray_addr)
    _print(f"Cluster resources: {ray.cluster_resources()}")

    # Optional: users can reference these env vars in their code to size tasks
    _print(f"USER_RESOURCES cpu={os.getenv('USER_CPUS')} gpu={os.getenv('USER_GPUS')} mem={os.getenv('USER_MEMORY')}")

    # Run user code (must exist in working_dir as user_main.py)
    try:
        import importlib.util, types, pathlib
        user_file = pathlib.Path("user_main.py")
        if not user_file.exists():
            raise FileNotFoundError("user_main.py not found in working_dir")

        spec = importlib.util.spec_from_file_location("user_main", str(user_file))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "main"):
            _print("Invoking user_main.main() ...")
            ret = mod.main()
            _print(f"user_main.main() returned: {ret}")
        else:
            _print("No main() found. Executing module top-level...")
        _print("DONE user code.")
    except Exception as e:
        _print("ERROR while running user code:")
        traceback.print_exc()
        sys.exit(1)
    finally:
        upload_results_if_any()

if __name__ == "__main__":
    main()

