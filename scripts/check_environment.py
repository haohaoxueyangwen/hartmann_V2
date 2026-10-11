"""Check imports and required entrypoints; never launch simulation or recovery."""
import importlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
deps={n:getattr(importlib.import_module(n),'__version__','OK') for n in ['numpy','scipy','matplotlib']}
for name in ['analysis.displacement_methods','analysis.zernike_reconstruction','analysis.optimization_common']:
    importlib.import_module(name)
required=['simulate.py','recover.py','config_simulate.py','config_recover.py']
missing=[f for f in required if not (root/f).exists()]
if missing:raise SystemExit('Missing files: '+', '.join(missing))
print(json.dumps({'dependencies':deps,'entrypoints':required,'scope':'Import check only; no simulation or physical validation.'},indent=2))
