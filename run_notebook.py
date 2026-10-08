"""Execute reversible_llm.ipynb in place, saving it after every cell.

    python run_notebook.py

Safe to re-run after a crash or power loss: finished runs replay their saved logs (results/<run>.json exists) and
an interrupted run continues from its last checkpoint (checkpoints/<run>_resume.pt). Live logs: results/<run>.log.
"""
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

here = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "reversible_llm.ipynb")
dst = sys.argv[2] if len(sys.argv) > 2 else src
cwd = os.path.dirname(os.path.abspath(src))
nb = nbformat.read(src, as_version=4)
client = NotebookClient(nb, timeout=None, kernel_name="python3", allow_errors=True, resources={"metadata": {"path": cwd}})
with client.setup_kernel():
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        t = time.time()
        client.execute_cell(cell, i)
        nbformat.write(nb, dst)
        err = [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]
        print(f"cell {i} done in {time.time() - t:.0f}s" + (f"  ERROR {err[0]['ename']}: {err[0]['evalue'][:200]}" if err else ""),
              flush=True)
nbformat.write(nb, dst)
print("notebook finished", flush=True)
