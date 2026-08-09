import os
import subprocess


def run(cmd, cwd=None, env=None, dry_run=False):
    """Run cmd (a list of argv strings), streaming output live.

    Inherits the calling process's environment (so the active conda env's
    binaries resolve normally); `env` supplies additive overrides, e.g. for
    CUDA_VISIBLE_DEVICES.
    """
    if dry_run:
        where = f" (cwd={cwd})" if cwd else ""
        print(f"[dry_run] would run: {' '.join(str(c) for c in cmd)}{where}")
        if env:
            print(f"[dry_run] env overrides: {env}")
        return

    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    subprocess.run(cmd, cwd=cwd, env=full_env, check=True)
