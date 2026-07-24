#!/usr/bin/env python3
"""Fetch top competitor generators from Hippius Hub for analysis."""

import os
from pathlib import Path
from hippius_hub import snapshot_download

# Top 5 competitors from the dashboard
competitors = [
    ("jan/cascade9", "c2d0d6b1", "cascade9-king"),
    ("jan/cascade10", "28cdfdfe", "cascade10"),
    ("knsimon/longrange-sv-v1", "88f00112", "longrange-sv"),
    ("my-cascade-gen/t_smo_v2", "2a2847c6", "t-smo-v2"),
    ("haruto/v7", "89f7741b", "haruto-v7"),
]

base_dir = Path(__file__).parent

for repo_id, revision, local_name in competitors:
    local_dir = base_dir / local_name
    print(f"\n📥 Fetching {repo_id}@{revision} → {local_name}/")
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        print(f"✅ Downloaded {repo_id}")
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {e}")

print("\n✅ All competitors fetched!")
