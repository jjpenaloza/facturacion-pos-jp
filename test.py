import sys, importlib, pycaret, sklearn, pandas, numpy, sktime
from importlib.util import find_spec
import pycaret.containers.models.time_series as ts

print("PYTHON   :", sys.version)
print("pycaret  :", pycaret.__version__)
print("sklearn  :", sklearn.__version__)
print("sktime   :", sktime.__version__)
print("pandas   :", pandas.__version__)
print("numpy    :", numpy.__version__)
print("prophet? :", find_spec("prophet") is not None)
print("tiene ProphetPeriodPatched?:", hasattr(ts, "ProphetPeriodPatched"))
