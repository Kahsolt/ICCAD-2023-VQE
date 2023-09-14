#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/13

from utils import *

from qiskit.algorithms.minimum_eigensolvers import VQE, VQEResult, AdaptVQE, AdaptVQEResult, SamplingVQE, SamplingVQEResult, QAOA
from qiskit.algorithms.minimum_eigensolvers import MinimumEigensolver, NumPyMinimumEigensolver, NumPyMinimumEigensolverResult
from qiskit_aer.primitives import Estimator
from qiskit_nature.second_q.problems import EigenstateResult, ElectronicStructureResult
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library.ansatzes import UCC, UCCSD, PUCCD, SUCCD, CHC, UVCC, UVCCSD
from qiskit_nature.second_q.circuit.library.initial_states import FermionicGaussianState, HartreeFock, SlaterDeterminant, VSCF

Result = Union[NumPyMinimumEigensolverResult, VQEResult, AdaptVQEResult, SamplingVQEResult]


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

fxs = []

def optim_callback(args, iter:int, params:Union[List[float], np.ndarray], fx:float, metadata:Dict[str, Any]):
  print(f'>> [{iter} / {args.maxiter}] fx = {fx}')
  if iter == 0: fxs.clear()
  fxs.append(fx)


@timer
def solver_vqe(args, ctx:Context) -> Tuple[float, Circuit]:
  mapper = JordanWignerMapper()
  ansatz = UCCSD(
    ctx.mol.num_spatial_orbitals,   # 6
    ctx.mol.num_particles,          # (4, 4)
    mapper,
    initial_state=HartreeFock(
      ctx.mol.num_spatial_orbitals,
      ctx.mol.num_particles,
      mapper,
    ),
    reps=1,
  )
  noise_model = load_noise_file(args.N) if args.N else None
  estimator = Estimator(
    backend_options = {
      'method': 'statevector',
      'device': 'CPU',
      'noise_model': noise_model,
    },
    transpile_options = {
      'seed_transpiler': args.seed,
    },
    run_options = {
      'shots': shots,
      'seed': args.seed,
    },
  )
  optim_cls = OPTIMZERS[args.O]
  optimizer = optim_cls(args, ansatz) if args.O == 'qnspsa' else optim_cls(args)

  solver = VQE(estimator, ansatz, optimizer, callback=partial(optim_callback, args))
  solver.initial_point = [0.0] * ansatz.num_parameters
  energy = GroundStateEigensolver_solve(args, solver, ctx)
  return energy, ansatz


@timer
def solver_adavqe(args, ctx:Context) -> Tuple[float, Circuit]:
  ansatz = None
  solver = AdaptVQE(VQE)
  energy = GroundStateEigensolver_solve(args, solver, ctx)
  return energy, ansatz


@timer
def solver_svqe(args, ctx:Context) -> Tuple[float, Circuit]:
  ansatz = None
  solver = SamplingVQE
  energy = GroundStateEigensolver_solve(args, solver, ctx)
  return energy, ansatz


@timer
def solver_qaoa(args, ctx:Context) -> Tuple[float, Circuit]:
  ansatz = None
  solver = QAOA
  energy = GroundStateEigensolver_solve(args, solver, ctx)
  return energy, ansatz
