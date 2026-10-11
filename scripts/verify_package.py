#!/usr/bin/env python
"""Verify the formal package file hashes without network or optional dependencies."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def verify():
    manifest=json.loads((ROOT/"FILELIST.json").read_text(encoding="utf-8"))
    failed=[]
    for row in manifest["files"]:
        p=ROOT/row["path"]
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row["sha256"]:failed.append(row["path"])
    if failed:raise SystemExit("Hash verification failed: "+", ".join(failed))
    return {"status":"passed","version":manifest["version"],"files_verified":len(manifest["files"])}
if __name__=="__main__":print(json.dumps(verify()))
