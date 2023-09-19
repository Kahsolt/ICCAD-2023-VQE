#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/12

from __future__ import annotations
import warnings ; warnings.filterwarnings('ignore', category=DeprecationWarning)

import sys
import json
import random
import pickle as pkl
from time import time
from pathlib import Path
from functools import partial
from datetime import datetime
from argparse import ArgumentParser, Namespace
from traceback import print_exc
from typing import *

import numpy as np
import matplotlib.pyplot as plt

from qiskit import pulse, transpile
from qiskit.pulse import ScheduleBlock
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit as Circuit
from qiskit.algorithms.optimizers import *
from qiskit.providers.fake_provider import FakeBackend, FakeMontreal
from qiskit.utils import algorithm_globals
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Sampler
import qiskit_nature ; qiskit_nature.settings.use_pauli_sum_op = False
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.drivers.pyscfd.pyscfdriver import ElectronicStructureProblem
from qiskit_nature.second_q.operators.fermionic_op import FermionicOp
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter

Problem = ElectronicStructureProblem
Hamiltonian = SparsePauliOp
Context = NamedTuple('Context', [('mol', Union[Problem, Namespace]), ('ham', Hamiltonian)])

BASE_PATH  = Path(__file__).parent.absolute()
REPO_PATH  = BASE_PATH / 'QC-Contest-Demo'
NOISE_PATH = REPO_PATH / 'NoiseModel'
HAM_FILE   = REPO_PATH / 'Hamiltonian' / 'OHhamiltonian.txt'
SEED_FILE  = REPO_PATH / 'algorithm_seeds' / 'requiredseeds.txt'
LOG_PATH   = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)

# https://qiskit.org/ecosystem/aer/stubs/qiskit_aer.AerSimulator.html
SIMULATORS = ['automatic', 'statevector', 'density_matrix', 'stabilizer', 'matrix_product_state', 'extended_stabilizer', 'unitary', 'superop']
# https://qiskit.org/documentation/stubs/qiskit.algorithms.optimizers.html
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
# https://qiskit.org/ecosystem/nature/apidocs/qiskit_nature.second_q.circuit.library.html
ANSATZS = [
  'uccs',
  'uccd',
  'uccsd',
  'uccst',
  'uccdt',
  'uccsdt',

  'uvccs',
  'uvccd',
  'uvccsd',
  'uvccst',
  'uvccdt',
  'uvccsdt',

  'puccd',
  'succd',

  'chccs',
  'chccd',
  'chccsd',
]

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

def gc_everything():
  import gc, os, psutil

  pid = os.getpid()
  mem = psutil.Process(pid).memory_info()
  print(f'[Mem] rss: {mem.rss/2**30:.3f} GB, vms: {mem.vms/2**30:.3f} GB')
  gc.collect()
  mem = psutil.Process(pid).memory_info()
  print(f'[Mem-GC] rss: {mem.rss/2**30:.3f} GB, vms: {mem.vms/2**30:.3f} GB')


def load_json(fp:Path, default:Any=dict) -> Dict:
  if not fp.exists():
    assert isinstance(default, Callable), '"default" should be a callable'
    return default()
  with open(fp, 'r', encoding='utf-8') as fh:
    return json.load(fh)

def save_json(data:Any, fp:Path):
  def _cvt(v:Any) -> Any:
    if isinstance(v, Path): return str(v)
    if isinstance(v, datetime): return v.isoformat()
    else: return v

  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, indent=2, ensure_ascii=False, default=_cvt)


def load_ham_file(fp:str) -> SparsePauliOp:
  from re import compile as Regex
  from copy import deepcopy

  def parse_chemiq(fp):
    ''' {'': -54.115564, 'X0 X1 X3 X4': -0.004611} '''

    R_QUBIT = Regex('([XYZI])(\d+)')

    with open(fp, 'r', encoding='utf-8') as fh:
      terms: Dict[str, float] = json.load(fh)

    n_qubit = 12
    string_proto = ['I'] * n_qubit

    if 'sanity check':
      max_idx = -1
      for qubits in terms:
        if not qubits: continue   # ignore identity ''
        for qubit in qubits.split(' '):
          gate, idx = R_QUBIT.findall(qubit)[0]
          max_idx = max(max_idx, int(idx))
      if max_idx != n_qubit - 1:
        print(f'warn: max qubit index ({max_idx}) != n_qubit - 1 ({n_qubit - 1}) ')

    def transform(qubits:str) -> str:
      ret = deepcopy(string_proto)
      if qubits:     # identity '' => 'III..'
        for qubit in qubits.split(' '):
          gate, idx = R_QUBIT.findall(qubit)[0]
          ret[int(idx)] = gate
      return ''.join(ret)
    
    return {transform(qubits): coeff for qubits, coeff in terms.items()}

  def parse_qiskit(fp):
    ''' one term per line: - 0.21924192058109332 * IIIIIIYZZYII '''

    R_LINE = Regex('([+-]?) *([\d\.]+) *\* *([IXYZ]+)')

    with open(fp, 'r', encoding='utf-8') as fh:
      lines = fh.read().strip().split('\n')

    terms = {}
    for line in lines:
      sign, coeff, string = R_LINE.findall(line.strip())[0]
      terms[string] = float(coeff) * (-1 if sign == '-' else 1)
    return terms

  try: terms = parse_qiskit(fp)
  except:
    try: terms = parse_chemiq(fp)
    except:
      print_exc()
      exit(-1)

  strings = list(terms.keys())    # 'IIIIIIYZZYII'
  coeffs = list(terms.values())   # -0.21924192058109332
  return SparsePauliOp(strings, coeffs)

def load_noise_file(name:str) -> NoiseModel:
  with open(NOISE_PATH / f'fake{name}.pkl', 'rb') as fh:
    noise_model_data = pkl.load(fh)
  return NoiseModel().from_dict(noise_model_data)


@timer
def get_context(args) -> Context:
  # requires PySCF (on Windows you cannot run this!)
  is_run = args.H == 'run'

  if is_run:
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
  else:
    print('<< make a dummy problem on Windows :)')
    problem = Namespace     # make a fake object
    problem.num_spatial_orbitals = 6
    problem.num_particles = (4, 4)
    problem.nuclear_repulsion_energy = 4.36537496654537

  print('[mol]')
  print('   n_spatial_orbitals:',       problem.num_spatial_orbitals)       # 6
  print('   n_particles:',              problem.num_particles)              # (4, 4)
  print('   nuclear_repulsion_energy:', problem.nuclear_repulsion_energy)   # 4.36537496654537

  if is_run:
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
      qubit_op: SparsePauliOp = mapper.map(fermi_op)
    else:
      converter = QubitConverter(mapper=mapper, two_qubit_reduction=True, sort_operators=True)
      qubit_op: SparsePauliOp = converter.convert(fermi_op)   # pauli
    print('   n_qubits:', qubit_op.num_qubits)      # 12
    pauli_op: SparsePauliOp = qubit_op.primitive
  else:
    pauli_op = load_ham_file(args.H)

  print('   n_ham_dim:', pauli_op.dim)            # (4096, 4096)
  print('   n_ham_terms:', pauli_op.size)         # 631
  pauli_op = pauli_op.simplify(args.thresh)
  pauli_op = pauli_op.chop(args.thresh)
  print('   n_ham_terms (trim):', pauli_op.size)  # <=631

  # FIXME: how to utilize this technique?
  pauli_op.group_commuting(qubit_wise=True)       # 136

  return Context(problem, pauli_op)


@timer
def run_pulse(args, ansatz:Circuit) -> Tuple[ScheduleBlock, Circuit]:
  name = f'Fake{args.S}'
  system_model: FakeBackend = globals()[name]()
  fp = LOG_PATH / f'{name}.json'
  if not fp.exists(): save_json(system_model.properties().to_dict(), fp)

  transpiled_ansatz = transpile(ansatz, backend=system_model)
  with pulse.build(system_model) as schedule:
    pulse.call(transpiled_ansatz)
  return schedule, transpiled_ansatz
