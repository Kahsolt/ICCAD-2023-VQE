#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/12

from __future__ import annotations
import warnings ; warnings.filterwarnings('ignore')

import random
import numpy as np
from typing import *

from qiskit.utils import algorithm_globals


# NOTE: fixed as the contest required
shots = 6000

def seed_everything(seed:int):
  random.seed(seed)
  np.random.seed(seed)
  algorithm_globals.random_seed = seed
