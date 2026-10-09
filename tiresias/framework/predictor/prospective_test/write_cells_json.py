"""Write cells_prosp.json (the cell list the GPU harness reads) from the 24 cell definitions (not from any table, so a static-analysis refusal cannot drop a cell from measurement)."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
import prosp_lib as L, cells_prosp as V
hw = L.UP.X.load_hardware(L.UP.X.read_text(L.UP.X.GROUND_TRUTH))
cells = [{k: v for k, v in c.items() if k != '_dev'} for c in V.define_cells(hw)]
(HERE / 'cells_prosp.json').write_text(json.dumps(dict(cells=sorted(cells, key=lambda c: c['cell_id'])), indent=1, sort_keys=True) + '\n'); print(len(cells), 'cells')
