if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import dezero.functions as F
import dezero.layers as L
from dezero.models import MLP
from dezero import optimizers
import dezero.datasets
from dezero import Variable, DataLoader, no_grad

