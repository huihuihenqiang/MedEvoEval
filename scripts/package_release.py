#!/usr/bin/env python
"""Build a clean distributable ZIP using the verified release file list."""
import argparse,json,zipfile
from pathlib import Path
from verify_package import verify,ROOT
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--output",type=Path,default=ROOT/"dist/MedEvoEval_code.zip")
a=p.parse_args();verify();out=a.output.resolve()
out.parent.mkdir(parents=True,exist_ok=True)
manifest=json.loads((ROOT/"FILELIST.json").read_text(encoding="utf-8"))
files=[r["path"] for r in manifest["files"]]+["FILELIST.json"]
with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for rel in sorted(files):
        info=zipfile.ZipInfo("MedEvoEval_code/"+rel,date_time=(2026,10,10,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
        z.writestr(info,(ROOT/rel).read_bytes())
print(json.dumps({"archive":str(out),"files":len(files)}))
