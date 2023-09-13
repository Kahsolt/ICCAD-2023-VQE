#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/12

from __future__ import annotations
import warnings ; warnings.filterwarnings('ignore', category=DeprecationWarning)

import random
import numpy as np
import pickle as pkl
from time import time
from pathlib import Path
from functools import partial
from argparse import ArgumentParser
from typing import *

from qiskit import pulse, transpile
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp, SparsePauliOp
from qiskit.circuit import QuantumCircuit as Circuit
from qiskit.algorithms.optimizers import *
from qiskit.providers.fake_provider import FakeMontreal
from qiskit.utils import algorithm_globals
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Sampler
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.drivers.pyscfd.pyscfdriver import ElectronicStructureProblem
from qiskit_nature.second_q.operators.fermionic_op import FermionicOp
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter

Molecular = ElectronicStructureProblem
Hamiltonian = PauliSumOp
Context = NamedTuple('Context', [('mol', Molecular), ('ham', Hamiltonian)])

BASE_PATH = Path(__file__).parent.absolute()
NOISE_MODEL_PATH = BASE_PATH / 'QC-Contest-Demo' / 'NoiseModel'
SEED_FILE = BASE_PATH / 'QC-Contest-Demo' / 'algorithm_seeds' / 'requiredseeds.txt'

OPTIMZERS = {
  'adam':    lambda args: ADAM           (maxiter=args.maxiter, tol=args.tol),
  'adam_ag': lambda args: ADAM           (maxiter=args.maxiter, tol=args.tol, amsgrad=True),
  'aqgd':    lambda args: AQGD           (maxiter=args.maxiter, tol=args.tol),
  'cg':      lambda args: CG             (maxiter=args.maxiter, tol=args.tol, disp=args.disp),
  'cobyla':  lambda args: COBYLA         (maxiter=args.maxiter, tol=args.tol, disp=args.disp),
  'gsls':    lambda args: GSLS           (maxiter=args.maxiter, disp=args.disp),
  'gd':      lambda args: GradientDescent(maxiter=args.maxiter, tol=args.tol),
  'lbfgsb':  lambda args: L_BFGS_B       (maxiter=args.maxiter, ftol=args.tol),
  'nm':      lambda args: NELDER_MEAD    (maxiter=args.maxiter, tol=args.tol, disp=args.disp),
  'nft':     lambda args: NFT            (maxiter=args.maxiter, disp=args.disp),
  'pbfgs':   lambda args: P_BFGS         (maxfun =args.maxiter, ftol=args.tol),
  'powell':  lambda args: POWELL         (maxiter=args.maxiter, tol=args.tol, disp=args.disp),
  'slsqp':   lambda args: SLSQP          (maxiter=args.maxiter, tol=args.tol, ftol=args.tol, disp=args.disp),
  'spsa':    lambda args: SPSA           (maxiter=args.maxiter),
  'qnspsa':  lambda args, ansatz: QNSPSA (maxiter=args.maxiter, fidelity=QNSPSA.get_fidelity(ansatz, Sampler())),
  'tnc':     lambda args: TNC            (maxiter=args.maxiter, tol=args.tol, ftol=args.tol, disp=args.disp),
  'crs':     lambda args: CRS            (max_evals=args.maxiter),
  'dl':      lambda args: DIRECT_L       (max_evals=args.maxiter),
  'dl_r':    lambda args: DIRECT_L_RAND  (max_evals=args.maxiter),
  'esch':    lambda args: ESCH           (max_evals=args.maxiter),
  'isres':   lambda args: ISRES          (max_evals=args.maxiter),
  'snobfit': lambda args: SNOBFIT        (max_evals=args.maxiter, verbose=args.disp),
  'bobyqa':  lambda args: BOBYQA         (maxiter=args.maxiter),
  'imfil':   lambda args: IMFIL          (maxiter=args.maxiter),
  'umda':    lambda args: UMDA           (maxiter=args.maxiter),
}

# NOTE: fixed as the contest required
shots = 6000
# NOTE: required seeds to average
with open(SEED_FILE, 'r', encoding='utf-8') as fh:
  lines = fh.read().strip().replace(',', '').split('\n')
  seeds = [int(seed) for seed in lines]


def timer(fn:Callable):
  def wrapper(*args, **kwargs):
    print(f'[Timer]: "{fn.__name__}" starts')
    start = time()
    r = fn(*args, **kwargs)
    end = time()
    print(f'[Timer]: "{fn.__name__}" took {end - start:.3f}s')
    return r
  return wrapper


def seed_everything(seed:int):
  random.seed(seed)
  np.random.seed(seed)
  algorithm_globals.random_seed = seed


@timer
def get_context(args) -> Context:
  driver = PySCFDriver(
    atom=[
      'O 0.0 0.0 0.0',
      'H 0.45 -0.1525 -0.8454',
    ],
    basis='sto3g',
    charge=1,     # OH-
    spin=0,
    unit=DistanceUnit.ANGSTROM,
  )
  problem = driver.run()

  print('[mol]')
  print('   n_spatial_orbitals:',       problem.num_spatial_orbitals)       # 6
  print('   n_particles:',              problem.num_particles)              # (4, 4)
  print('   nuclear_repulsion_energy:', problem.nuclear_repulsion_energy)   # 4.36537496654537

  print('[ham]')
  ham: ElectronicEnergy = problem.hamiltonian
  print('   ham.register_length:', ham.register_length)   # 6
  fermi_op: FermionicOp = ham.second_q_op()  # fermi
  print('   len(fermi_op):', len(fermi_op))               # 1860
  fermi_op = fermi_op.normal_order()
  print('   len(fermi_op_ordered):', len(fermi_op))       # 630
  print('   num_spin_orbitals:', fermi_op.num_spin_orbitals)  # 12

  mapper = JordanWignerMapper()
  if 'new API':
    qubit_op: PauliSumOp = mapper.map(fermi_op)
  else:
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=True, sort_operators=True)
    qubit_op: PauliSumOp = converter.convert(fermi_op)   # pauli
  print('   n_qubits:', qubit_op.num_qubits)      # 12

  pauli_op: SparsePauliOp = qubit_op.primitive
  print('   n_ham_dim:', pauli_op.dim)            # (4096, 4096)
  print('   n_ham_terms:', pauli_op.size)         # 631
  pauli_op = pauli_op.simplify(args.thresh)
  pauli_op = pauli_op.chop(args.thresh)
  print('   n_ham_terms (trim):', pauli_op.size)  # <=631

  # FIXME: how to utilize this technique?
  pauli_op.group_commuting(qubit_wise=True)       # 136

  return Context(problem, pauli_op)


def get_noise_model(name:str) -> NoiseModel:
  with open(NOISE_MODEL_PATH / f'fake{name}.pkl', 'rb') as fh:
    noise_model_data = pkl.load(fh)
  return NoiseModel().from_dict(noise_model_data)


@timer
def run_pulse(args, ansatz:Circuit) -> float:
  system_model = globals()[f'Fake{args.S}']()
  transpiled_ansatz = transpile(ansatz, backend=system_model)
  with pulse.build(system_model) as prog:
    pulse.call(transpiled_ansatz)
  return prog.duration
