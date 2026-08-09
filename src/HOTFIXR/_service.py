import os
import subprocess
import time

from .config import REPO_ROOT


def ensure_reward_service_running(port=5145, startup_timeout=180, dry_run=False):
    """Make sure services/all.py's FastAPI reward server is up on `port`,
    starting it as a detached background process if it isn't already
    running. Replaces having to separately `source run_services.sh` in
    another terminal before calling train_generator().
    """
    import requests

    def is_up():
        try:
            requests.get(f"http://localhost:{port}/openapi.json", timeout=2)
            return True
        except requests.exceptions.RequestException:
            return False

    if dry_run:
        if is_up():
            print(f"[dry_run] reward service already running on port {port}")
        else:
            print(f"[dry_run] would start reward service in the background: "
                  f"python services/all.py")
        return

    if is_up():
        return

    server_env = os.environ.copy()
    server_env.setdefault("SERVER_IP", "0.0.0.0")
    log_path = REPO_ROOT / "services" / "server.log"
    log_file = open(log_path, "a")
    subprocess.Popen(
        ["python", str(REPO_ROOT / "services" / "all.py")],
        cwd=REPO_ROOT,
        env=server_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if is_up():
            return
        time.sleep(2)
    raise RuntimeError(
        f"Reward service did not come up on port {port} within {startup_timeout}s; "
        f"check {log_path}"
    )


def stop_reward_service(port=5145, dry_run=False):
    """POST /end_service (services/all.py:97) to shut the reward server down."""
    if dry_run:
        print(f"[dry_run] would POST http://localhost:{port}/end_service")
        return

    import requests
    try:
        # The server kills its own process (os.kill(getpid(), SIGTERM)) as
        # part of handling this request, so the connection is often reset
        # before a response comes back — that's expected, not an error.
        requests.post(f"http://localhost:{port}/end_service", timeout=5)
    except requests.exceptions.RequestException:
        pass
