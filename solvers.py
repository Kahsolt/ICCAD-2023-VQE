#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/13

from utils import *

from qiskit.algorithms.minimum_eigensolvers import VQE, AdaptVQE, SamplingVQE, QAOA
from qiskit.algorithms.minimum_eigensolvers import MinimumEigensolver, NumPyMinimumEigensolver


def run_solver(args, name:str, ctx:Context) -> Union[float, Tuple[float, Circuit]]:
  return globals()[f'solver_{name}'](args, ctx)


def GroundStateEigensolver_solve(args, solver:MinimumEigensolver, ctx:Context) -> float:
  from qiskit_nature.second_q.problems.eigenstate_result import EigenstateResult
  from qiskit_nature.second_q.algorithms.ground_state_solvers import GroundStateEigensolver

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

def optim_callback(args, iter:int, params:List[float], fx:float, metadata:Options):
  print(f'>> [{iter} / {args.maxiter}] fx = {fx}')
  if iter == 1: fxs.clear()
  fxs.append(fx)


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
