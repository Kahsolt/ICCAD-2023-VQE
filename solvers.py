#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/13

from utils import *

from qiskit.algorithms.minimum_eigensolvers import VQE, VQEResult, AdaptVQE, AdaptVQEResult, SamplingVQE, SamplingVQEResult, QAOA
from qiskit.algorithms.minimum_eigensolvers import MinimumEigensolver, NumPyMinimumEigensolver, NumPyMinimumEigensolverResult
from qiskit_aer.primitives import Estimator
from qiskit_nature.second_q.problems import EigenstateResult, ElectronicStructureResult
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint, VSCFInitialPoint
from qiskit_nature.second_q.circuit.library.ansatzes import UCC, UCCSD, PUCCD, SUCCD, CHC, UVCC, UVCCSD
from qiskit_nature.second_q.circuit.library.ansatzes.utils import generate_vibration_excitations
from qiskit_nature.second_q.circuit.library.initial_states import HartreeFock, FermionicGaussianState, SlaterDeterminant, VSCF

Result = Union[NumPyMinimumEigensolverResult, VQEResult, AdaptVQEResult, SamplingVQEResult]
Params = np.ndarray
Options = Dict[str, Any]


def run_solver(args, name:str, ctx:Context) -> Union[float, Tuple[float, Circuit]]:
  return globals()[f'solver_{name}'](args, ctx)


def GroundStateEigensolver_solve(args, solver:MinimumEigensolver, ctx:Context) -> float:
  assert GroundStateEigensolver.solve
  raw_res: Result = solver.compute_minimum_eigenvalue(ctx.ham)
  raw_res = EigenstateResult.from_result(raw_res)
  if isinstance(ctx.mol, Problem):    # PySCFDriver generated
    res = Problem.interpret(ctx.mol, raw_res)
    if args.disp: print(res)
    gs_ene = res.total_energies.item()
  else:                               # fake
    if args.disp: print(raw_res)
    gs_ene = raw_res.groundenergy + ctx.mol.nuclear_repulsion_energy
  return gs_ene


# ↓↓↓ classical solvers ↓↓↓

@timer
def solver_const(args, ctx:Context) -> float:
  return -74.3871462681872


@timer
def solver_numpy(args, ctx:Context) -> float:
  solver = NumPyMinimumEigensolver()
  return GroundStateEigensolver_solve(args, solver, ctx)


# ↓↓↓ quantum solvers ↓↓↓

def _get_primitive_options(args) -> Options:
  return {
    'backend_options': {     # the backend hardware
      'method': args.simulator,
      'device': args.device,
      'noise_model': load_noise_file(args.N) if args.N else None,
    },
    'transpile_options': {   # compile the circuit to backend hardware
      'seed_transpiler': args.seed,
    },
    'run_options': {         # run the measurement
      'shots': args.shots,
      'seed': args.seed,
    },
  }

def get_estimator(args) -> Estimator:
  ''' Estimator is a runtime primitive, handling multiple circuits '''
  # the virtual/physical machine env to run the circuit
  return Estimator(**_get_primitive_options(args))

def get_sampler(args) -> Sampler:
  ''' Sampler is a light-weight runtime primitive, handling one circuit '''
  return Sampler(**_get_primitive_options(args))


def get_ansatz(args, ctx:Context) -> Tuple[Circuit, Params]:
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
  elif args.ansatz.startswith('uvcc'):
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

  n_params = ansatz.num_parameters
  if   args.init == 'none':  init = None
  elif args.init == 'zeros': init = np.zeros(shape=[n_params])
  elif args.init == 'randu': init = np.random.uniform(low=-np.pi/4, high=np.pi/4, size=[n_params])
  elif args.init == 'randn': init = np.random.normal(loc=0, scale=0.2, size=[n_params])
  return ansatz, init


fxs = []

def optim_callback(args, iter:int, params:List[float], fx:float, metadata:Options):
  print(f'>> [{iter} / {args.maxiter}] fx = {fx}')
  if iter == 1: fxs.clear()
  fxs.append(fx)

def get_optimizer(args, ansatz:Circuit) -> Optimizer:
  optim_cls = OPTIMZERS[args.O]
  optimizer = optim_cls(args, ansatz) if args.O == 'qnspsa' else optim_cls(args)
  return optimizer


@timer
def solver_vqe(args, ctx:Context) -> Tuple[float, Circuit]:
  estimator = get_estimator(args)
  ansatz, init = get_ansatz(args, ctx)
  optimizer = get_optimizer(args, ansatz)
  solver = VQE(estimator, ansatz, optimizer, callback=partial(optim_callback, args))
  solver.initial_point = init
  energy = GroundStateEigensolver_solve(args, solver, ctx)
  return energy, ansatz


@timer
def solver_adavqe(args, ctx:Context) -> Tuple[float, Circuit]:
  estimator = get_estimator(args)
  ansatz, init = get_ansatz(args, ctx)
  optimizer = get_optimizer(args, ansatz)
  vqe = VQE(estimator, ansatz, optimizer, callback=partial(optim_callback, args))
  solver = AdaptVQE(vqe)
  solver.initial_point = init
  energy = GroundStateEigensolver_solve(args, solver, ctx)
  return energy, ansatz


@timer
def solver_svqe(args, ctx:Context) -> Tuple[float, Circuit]:
  sampler = get_sampler(args)
  ansatz, init = get_ansatz(args, ctx)
  optimizer = get_optimizer(args, ansatz)
  solver = SamplingVQE(sampler, ansatz, optimizer, callback=partial(optim_callback, args))
  solver.initial_point = init
  energy = GroundStateEigensolver_solve(args, solver, ctx)
  return energy, ansatz


@timer
def solver_qaoa(args, ctx:Context) -> Tuple[float, Circuit]:
  sampler = get_sampler(args)
  optimizer = get_optimizer(args, None)
  solver = QAOA(sampler, optimizer, reps=args.reps, callback=partial(optim_callback, args))
  energy = GroundStateEigensolver_solve(args, solver, ctx)
  return energy, solver.ansatz
