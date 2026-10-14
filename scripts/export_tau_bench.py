"""Export data from the original tau-bench repository into static JSON for the playground.

Usage:
    git clone https://github.com/sierra-research/tau-bench /path/to/tau-bench
    python scripts/export_tau_bench.py /path/to/tau-bench

Writes public/data/{retail,airline}/ with:
    db.json            initial database (key order and int/float types preserved)
    tasks_<split>.json task instructions, ground-truth actions and required outputs
    tools.json         tool schemas exactly as sent to the agent, plus Python signatures
    wiki.md            the domain policy (the agent's system prompt)
and public/data/manifest.json.

Only the environment modules are imported; `litellm` is stubbed so no LLM SDK is needed.
"""

import inspect
import json
import os
import subprocess
import sys
import types
from datetime import datetime, timezone

DOMAINS = {
    "retail": {
        "splits": {
            "test": ("tau_bench.envs.retail.tasks_test", "TASKS_TEST"),
            "train": ("tau_bench.envs.retail.tasks_train", "TASKS_TRAIN"),
            "dev": ("tau_bench.envs.retail.tasks_dev", "TASKS_DEV"),
        },
    },
    "airline": {
        "splits": {
            "test": ("tau_bench.envs.airline.tasks_test", "TASKS"),
        },
    },
}

# Same for both domains (see MockRetailDomainEnv / MockAirlineDomainEnv).
TERMINATE_TOOLS = ["transfer_to_human_agents"]


def dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        # Python's default float repr keeps "46.0" distinguishable from "46".
        json.dump(obj, f, separators=(",", ":"))


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = os.path.abspath(sys.argv[1])
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public", "data")
    out_root = os.path.normpath(out_root)

    # tau_bench.envs.user imports litellm at module load; the environments never call it.
    sys.modules["litellm"] = types.SimpleNamespace(completion=None, provider_list=[])
    sys.path.insert(0, src)
    import importlib

    try:
        commit = subprocess.check_output(["git", "-C", src, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None

    manifest = {
        "source": "https://github.com/sierra-research/tau-bench",
        "commit": commit,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "terminate_tools": TERMINATE_TOOLS,
        "domains": {},
    }

    for domain, spec in DOMAINS.items():
        out_dir = os.path.join(out_root, domain)
        os.makedirs(out_dir, exist_ok=True)

        data_mod = importlib.import_module(f"tau_bench.envs.{domain}.data")
        dump(os.path.join(out_dir, "db.json"), data_mod.load_data())

        tools_mod = importlib.import_module(f"tau_bench.envs.{domain}.tools")
        tools = []
        for cls in tools_mod.ALL_TOOLS:
            params = [p for p in inspect.signature(cls.invoke).parameters if p != "data"]
            tools.append(
                {
                    "class_name": cls.__name__,
                    "qualified_name": f"{cls.__module__}.{cls.invoke.__qualname__}",
                    "params": params,
                    "info": cls.get_info(),
                }
            )
        dump(os.path.join(out_dir, "tools.json"), tools)

        wiki_mod = importlib.import_module(f"tau_bench.envs.{domain}.wiki")
        with open(os.path.join(out_dir, "wiki.md"), "w", encoding="utf-8") as f:
            f.write(wiki_mod.WIKI)

        split_counts = {}
        for split, (module, attr) in spec["splits"].items():
            tasks = getattr(importlib.import_module(module), attr)
            dump(
                os.path.join(out_dir, f"tasks_{split}.json"),
                [
                    {
                        "user_id": t.user_id,
                        "instruction": t.instruction,
                        "actions": [{"name": a.name, "kwargs": a.kwargs} for a in t.actions],
                        "outputs": t.outputs,
                    }
                    for t in tasks
                ],
            )
            split_counts[split] = len(tasks)
        manifest["domains"][domain] = {"splits": split_counts, "tools": len(tools)}
        print(f"{domain}: {split_counts}, {len(tools)} tools")

    with open(os.path.join(out_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
