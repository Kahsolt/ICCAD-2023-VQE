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
from qiskit.primitives import EstimatorResult
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit import QuantumCircuit as Circuit
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.algorithms.optimizers import *
from qiskit.algorithms.minimum_eigensolvers import VQEResult, AdaptVQEResult, SamplingVQEResult, NumPyMinimumEigensolverResult
from qiskit.providers import JobV1 as Job
from qiskit.providers.fake_provider import FakeBackend, FakeCairo, FakeKolkata, FakeMontreal
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.utils.mitigation import CompleteMeasFitter
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator, Sampler
from qiskit_aer.noise import NoiseModel
import qiskit_nature ; qiskit_nature.settings.use_pauli_sum_op = False
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers.pyscfd.pyscfdriver import PySCFDriver, ElectronicStructureProblem
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import PolynomialTensor, ElectronicIntegrals
from qiskit_nature.second_q.operators.fermionic_op import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint, MP2InitialPoint, VSCFInitialPoint
from qiskit_nature.second_q.circuit.library.ansatzes import UCC, UCCSD, PUCCD, SUCCD, CHC, UVCC, UVCCSD
from qiskit_nature.second_q.circuit.library.ansatzes.utils import generate_vibration_excitations
from qiskit_nature.second_q.circuit.library.initial_states import HartreeFock, FermionicGaussianState, SlaterDeterminant, VSCF

Problem = ElectronicStructureProblem
Hamiltonian = SparsePauliOp
Context = NamedTuple('Context', [('mol', Problem), ('ham', Hamiltonian)])
Result = Union[NumPyMinimumEigensolverResult, VQEResult, AdaptVQEResult, SamplingVQEResult]
Params = np.ndarray
Options = Dict[str, Any]
QubitMapping = Dict[int, int]

BASE_PATH  = Path(__file__).parent.absolute()
REPO_PATH  = BASE_PATH / 'QC-Contest-Demo'
NOISE_PATH = REPO_PATH / 'NoiseModel'
HAM_FILE   = REPO_PATH / 'Hamiltonian' / 'OHhamiltonian.txt'
SEED_FILE  = REPO_PATH / 'algorithm_seeds' / 'requiredseeds.txt'
INTG_FILE  = BASE_PATH / 'integrals.pkl'
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

  'chcs',
  'chcd',
  'chcsd',
]
# https://qiskit.org/ecosystem/nature/apidocs/qiskit_nature.second_q.algorithms.initial_points.html
INITS = [
  'auto',
  'hf',
  'mp2',
  'vscf',
  'none',
  'zeros',
  'noise',
  'randu',
  'randn',
]

# NOTE: required seeds to average
with open(SEED_FILE, 'r', encoding='utf-8') as fh:
  lines = fh.read().strip().replace(',', '').split('\n')
  seeds = [int(seed) for seed in lines]

log = print

def timer(fn:Callable):
  def wrapper(*args, **kwargs):
    log(f'[Timer]: "{fn.__name__}" starts')
    start = time()
    r = fn(*args, **kwargs)
    end = time()
    log(f'[Timer]: "{fn.__name__}" took {end - start:.3f}s')
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

    if not INTG_FILE.exists():
      log('<< cache the integrals on Linux :)')
      integrals = problem.hamiltonian.electronic_integrals
      tensor_to_numpy = lambda x: { k: v.array for k, v in x._data.items() }
      integral_parts = {
        'alpha':      tensor_to_numpy(integrals.alpha),
        'beta':       tensor_to_numpy(integrals.beta),
        'beta_alpha': tensor_to_numpy(integrals.beta_alpha),
      }
      with open(INTG_FILE, 'wb') as fh:
        pkl.dump(integral_parts, fh)
  else:
    log('<< make a dummy problem on Windows :)')
    with open(INTG_FILE, 'rb') as fh:
      raw_integral_parts: Dict[str, Dict[str, np.ndarray]] = pkl.load(fh)
    integral_parts = { k: PolynomialTensor(v) for k, v in raw_integral_parts.items() }
    integrals = ElectronicIntegrals(**integral_parts)
    energy = ElectronicEnergy(integrals, constants={
      'nuclear_repulsion_energy': 4.36537496654537,
    })
    problem = ElectronicStructureProblem(energy)
    problem.num_spatial_orbitals = 6
    problem.num_particles = (4, 4)

  log('[mol]')
  log('   n_spatial_orbitals:',       problem.num_spatial_orbitals)       # 6
  log('   n_particles:',              problem.num_particles)              # (4, 4)
  log('   nuclear_repulsion_energy:', problem.nuclear_repulsion_energy)   # 4.36537496654537

  if is_run:
    log('[ham]')
    ham: ElectronicEnergy = problem.hamiltonian
    log('   ham.register_length:', ham.register_length)       # 6
    fermi_op: FermionicOp = ham.second_q_op()  # fermi
    log('   len(fermi_op):', len(fermi_op))                   # 1860
    fermi_op = fermi_op.normal_order()
    log('   len(fermi_op_ordered):', len(fermi_op))           # 630
    log('   num_spin_orbitals:', fermi_op.num_spin_orbitals)  # 12

    mapper = JordanWignerMapper()
    if 'new API':
      pauli_op: SparsePauliOp = mapper.map(fermi_op)
    else:
      converter = QubitConverter(mapper=mapper, two_qubit_reduction=True, sort_operators=True)
      pauli_op: SparsePauliOp = converter.convert(fermi_op)   # pauli
    log('   n_qubits:', pauli_op.num_qubits)      # 12
  else:
    pauli_op = load_ham_file(args.H)

  log('   n_ham_dim:', pauli_op.dim)              # (4096, 4096)
  log('   n_ham_terms:', pauli_op.size)           # 631
  pauli_op = pauli_op.simplify(args.thresh)
  pauli_op = pauli_op.chop(args.thresh)
  log('   n_ham_terms (trim):', pauli_op.size)    # <=631

  # FIXME: how to utilize this technique?
  pauli_op.group_commuting(qubit_wise=True)       # 136

  return Context(problem, pauli_op)


def _get_primitive_options(args, noisy:bool=True, transpile:bool=True) -> Options:
  options = {
    'backend_options': {     # the backend hardware
      'method': args.simulator,
      'device': args.device,
      'noise_model': load_noise_file(args.N) if (noisy and args.N) else None,
    },
    'transpile_options': {   # compile the circuit to backend hardware
      'seed_transpiler': args.seed,
    },
    'run_options': {         # run the measurement
      'shots': args.shots,
      'seed': args.seed,
    },
    'skip_transpilation': not transpile,
  }
  return options

def get_estimator(args, options:Options=None) -> Estimator:
  ''' Estimator is a runtime primitive, handling multiple circuits '''
  # the virtual/physical machine env to run the circuit
  return Estimator(**(options or _get_primitive_options(args)))

def get_sampler(args, options:Options=None) -> Sampler:
  ''' Sampler is a light-weight runtime primitive, handling one circuit '''
  return Sampler(**(options or _get_primitive_options(args)))


def get_backend(args) -> FakeBackend:
  name = f'Fake{args.S}'
  system_model: FakeBackend = globals()[name]()
  if 'save device info':
    fp = LOG_PATH / f'{name}.json'
    if not fp.exists(): save_json(system_model.properties().to_dict(), fp)
  return system_model


def get_ansatz(args, ctx:Context) -> Tuple[Circuit, Params]:
  # ansatz
  mapper = JordanWignerMapper()
  if args.ansatz.startswith('ucc'):
    ucc_type = args.ansatz[3:]
    ansatz = UCC(
      ctx.mol.num_spatial_orbitals,   # 6
      ctx.mol.num_particles,          # (4, 4)
      ucc_type,
      mapper,
      reps=args.reps,
      initial_state=HartreeFock(
        ctx.mol.num_spatial_orbitals,
        ctx.mol.num_particles,
        mapper,
      ),
    )
  elif args.ansatz.startswith('uvcc'):    # NOTE: this does not work due to not support JordanWignerMapper :(
    uvcc_type = args.ansatz[4:]
    ansatz = UVCC(
      ctx.mol.num_particles,          # (4, 4)
      uvcc_type,
      mapper,
      reps=args.reps,
      initial_state=VSCF(
        ctx.mol.num_particles,
        mapper,
      ),
    )
  elif args.ansatz == 'puccd':    # only for single-spin system
    ansatz = PUCCD(
      ctx.mol.num_spatial_orbitals,   # 6
      ctx.mol.num_particles,          # (4, 4)
      mapper,
      reps=args.reps,
      initial_state=HartreeFock(
        ctx.mol.num_spatial_orbitals,
        ctx.mol.num_particles,
        mapper,
      ),
    )
  elif args.ansatz == 'succd':    # only for single-spin system
    ansatz = SUCCD(
      ctx.mol.num_spatial_orbitals,   # 6
      ctx.mol.num_particles,          # (4, 4)
      mapper,
      reps=args.reps,
      initial_state=HartreeFock(
        ctx.mol.num_spatial_orbitals,
        ctx.mol.num_particles,
        mapper,
      ),
    )
  elif args.ansatz.startswith('chc'):      # approx UCC with less CNOT
    chc_type = args.ansatz[3:]
    excitations = [
      *(generate_vibration_excitations(1, ctx.mol.num_particles) if 's' in chc_type else []),
      *(generate_vibration_excitations(2, ctx.mol.num_particles) if 'd' in chc_type else []),
    ]
    ansatz = CHC(
      ctx.ham.num_qubits,
      excitations,
      reps=args.reps,
      initial_state=HartreeFock(
        ctx.mol.num_spatial_orbitals,
        ctx.mol.num_particles,
        mapper,
      )
    )
  else: raise ValueError(f'unknown ansatz {args.ansatz}')

  # override init
  if args.init == 'auto':
    if   isinstance(ansatz, UCC):  args.init = 'hf'
    elif isinstance(ansatz, UVCC): args.init = 'vscf'
    else: args.init = 'zeros'

  # init params point with theories (not the init |state>)
  init = None
  if   args.init == 'hf':
    hf_initial_point = HFInitialPoint()
    hf_initial_point.ansatz = ansatz
    hf_initial_point.problem = ctx.mol
    init = hf_initial_point.to_numpy_array()
  elif args.init == 'mp2':
    mp2_initial_point = MP2InitialPoint()
    mp2_initial_point.ansatz = ansatz
    mp2_initial_point.problem = ctx.mol
    init = mp2_initial_point.to_numpy_array()
  elif args.init == 'vscf':
    vscf_initial_point = VSCFInitialPoint()
    vscf_initial_point.ansatz = ansatz
    vscf_initial_point.problem = ctx.mol
    init = vscf_initial_point.to_numpy_array()
  # other init without theories
  n_params = ansatz.num_parameters
  if   args.init == 'none':  pass
  elif args.init == 'zeros': init = np.zeros(shape=[n_params])
  elif args.init == 'noise': init = np.random.uniform(low=-1e-5, high=1e-5, size=[n_params])
  elif args.init == 'randu': init = np.random.uniform(low=-np.pi/4, high=np.pi/4, size=[n_params])
  elif args.init == 'randn': init = np.random.normal(loc=0, scale=0.2, size=[n_params])
  return ansatz.decompose(), init

def _remove_idle_qwires(circ:Circuit) -> Circuit:
  # https://quantumcomputing.stackexchange.com/questions/25672/remove-inactive-qubits-from-qiskit-circuit
  from qiskit.converters import circuit_to_dag, dag_to_circuit
  from collections import OrderedDict
  dag = circuit_to_dag(circ)
  idle_wires = list(dag.idle_wires())
  for w in idle_wires:
    dag._remove_idle_wire(w)
    dag.qubits.remove(w)
  dag.qregs = OrderedDict()
  return dag_to_circuit(dag)


def get_optimizer(args, ansatz:Circuit) -> Optimizer:
  optim_cls = OPTIMZERS[args.O]
  optimizer = optim_cls(args, ansatz) if args.O == 'qnspsa' else optim_cls(args)
  return optimizer


@timer
def run_noisy_eval(args, ctx:Context, ansatz:Circuit) -> Tuple[float, Circuit, QubitMapping, int]:
  ''' real backend config noisy simulatoion '''

  # run circuit
  estimator = get_estimator(args, _get_primitive_options(args, noisy=True, transpile=True))
  job: Job = estimator.run(ansatz, ctx.ham)
  result: EstimatorResult = job.result()
  ene_gs_noiseless = result.experiments[0]['values'] + ctx.mol.nuclear_repulsion_energy
  print('>> run circuit:', ene_gs_noiseless)

  # run transpiled circuit
  system_model = get_backend(args)
  # NOTE: transpile does NOT conserve working num_qubit for UCC-like ansatz :(
  transpiled_circuit = transpile(ansatz, backend=system_model, seed_transpiler=args.seed, optimization_level=args.level)
  qubit_mapping: Dict[Qubit, int] = transpiled_circuit.layout.input_qubit_mapping
  transpiled_circuit.layout.initial_layout
  transpiled_circuit.layout.final_layout
  transpiled_circuit.layout.input_qubit_mapping
  mapping = { i: qbit.index for qbit, i in qubit_mapping.items() }  # original => transpiled
  ene_gs_noisy = run_transpiled_circuit(args, ctx, transpiled_circuit, mapping)
  print('>> run transpiled circuit:', ene_gs_noisy)

  duration = run_pulse(system_model, transpiled_circuit)
  return ene_gs_noisy, transpiled_circuit, mapping, duration


def run_pulse(system_model:FakeBackend, transpiled_circuit:Circuit) -> int:
  with pulse.build(system_model) as schedule:
    schedule: ScheduleBlock
    pulse.call(transpiled_circuit)
  return schedule.duration


def run_transpiled_circuit(args, ctx:Context, transpiled_circuit:Circuit, mapping:QubitMapping) -> float:
  ham = ctx.ham

  if 'align circuit and ham, apply qubit-index mapping':
    def permute(string: str) -> str:
      res = [None] * len(string)
      for i, j in enumerate(mapping):
        res[j] = string[i]
      return ''.join(res)

    n_qubits_ex = transpiled_circuit.num_qubits - ham.num_qubits
    Is = 'I' * n_qubits_ex
    paulis: List[Pauli] = [Pauli(permute(p.expand(Is).to_label())) for p in ham.paulis]
    ham_ex = Hamiltonian(paulis, ham.coeffs)
    assert ham_ex.num_qubits == transpiled_circuit.num_qubits

  estimator = get_estimator(args, _get_primitive_options(args, noisy=True, transpile=False))
  job: Job = estimator.run(transpiled_circuit, ham_ex)
  result: EstimatorResult = job.result()
  ene_gs = result.experiments[0]['values'] + ctx.mol.nuclear_repulsion_energy
  return ene_gs


def get_args():
  parser = ArgumentParser()
  # hardware
  parser.add_argument('-S', default='Montreal', choices=['Montreal'], help='system model (27 qubits)')
  parser.add_argument('-N', default='', choices=['', 'cairo', 'kolkata', 'montreal'], help='noise model data (27 qubits)')
  parser.add_argument('--simulator', default='automatic', choices=SIMULATORS, help='qiskit_aer simulator method')
  parser.add_argument('--shots', default=6000, help='shot-based simulator resource limit')
  parser.add_argument('--shots_mit', default=1000, help='error mitigation shots')
  parser.add_argument('--level', default=0, type=int, choices=[0, 1, 2, 3], help='transpile optimize level')
  parser.add_argument('--skip_pulse', action='store_true', help='skip run real backend noisy simulation')
  # ham
  parser.add_argument('-H', default='run', help=f'"run" requires PySCF, "txt" loads {HAM_FILE}; or filepath to your own ham, see fmt in `utils.load_ham_file()`')
  parser.add_argument('--thresh', default=1e-6, type=float, help='ham term trim amplitude thresh')
  # eigensolver
  parser.add_argument('-Y', default='numpy', choices=['numpy', 'const'], help='classical eigensolver (ref_val)')
  parser.add_argument('-X', default='vqe', choices=['vqe', 'adavqe', 'svqe', 'qaoa'], help='quantum eigensolver')
  # ansatz
  parser.add_argument('-A', '--ansatz', default='uccsd', choices=ANSATZS, help='ansatz model')
  parser.add_argument('-I', '--init',   default='auto', choices=INITS, help='ansatz param init')
  parser.add_argument('--reps', default=1, type=int, help='ansatz circuit n_repeats')
  # optim
  parser.add_argument('-O', default='cobyla', choices=OPTIMZERS.keys(), help='optim method')
  parser.add_argument('-T', '--maxiter', default=10, type=int, help='optim maxiter')
  parser.add_argument('--tol', default=1e-5, type=float, help='optim tolerance')
  parser.add_argument('--disp', action='store_true', help='optim show verbose result')
  # misc
  parser.add_argument('--seed', default=170, type=int, help='rand seed')
  parser.add_argument('--name', help='experiment name under log folder, default to time-string')
  args, _ = parser.parse_known_args()

  if args.H == 'txt': args.H = str(HAM_FILE)
  if args.H != 'run': assert Path(args.H).is_file(), f'-H {args.H!r} should be a valid file'

  seed_everything(args.seed)
  args.device = 'GPU' if 'GPU' in AerSimulator().available_devices() else 'CPU'

  return args


def get_eval_args():
  parser = ArgumentParser()
  # experiment log
  parser.add_argument('exp_dp', type=Path, help='path to exp folder, e.g. log/my_exp')
  # hardware
  parser.add_argument('-S', default='Montreal', choices=['Montreal'], help='system model (27 qubits)')
  parser.add_argument('-N', default='', choices=['', 'cairo', 'kolkata', 'montreal'], help='noise model data (27 qubits)')
  parser.add_argument('--simulator', default='automatic', choices=SIMULATORS, help='qiskit_aer simulator method')
  parser.add_argument('--shots', default=6000, help='shot-based simulator resource limit')
  # ham
  parser.add_argument('-H', default='txt', help=f'"run" requires PySCF, "txt" loads {HAM_FILE}; or filepath to your own ham, see fmt in `utils.load_ham_file()`')
  parser.add_argument('--thresh', default=1e-6, type=float, help='ham term trim amplitude thresh')
  # misc
  parser.add_argument('--seed', default=170, type=int, help='rand seed')
  args, _ = parser.parse_known_args()

  exp_dp = Path(args.exp_dp)
  assert exp_dp.is_dir(), '`exp_dp` must be a valid folder'
  qsam_fp = exp_dp / 'ansatz_t.qsam'
  assert qsam_fp.is_file(), f'missing qsam file: {qsam_fp}'
  layout_fp = exp_dp / 'ansatz_t.layout'
  assert layout_fp.is_file(), f'missing layout file: {layout_fp}'
  args.qsam_fp = qsam_fp
  args.layout_fp = layout_fp

  if args.H == 'txt': args.H = str(HAM_FILE)
  if args.H != 'run': assert Path(args.H).is_file(), f'-H {args.H!r} should be a valid file'

  seed_everything(args.seed)
  args.device = 'GPU' if 'GPU' in AerSimulator().available_devices() else 'CPU'

  return args
