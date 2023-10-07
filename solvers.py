#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/13

from utils import *

from qiskit.algorithms.minimum_eigensolvers import VQE, AdaptVQE, SamplingVQE, QAOA
from qiskit.algorithms.minimum_eigensolvers import MinimumEigensolver, NumPyMinimumEigensolver


def run_solver(args, name:str, ctx:Context) -> Union[float, Tuple[float, Tuple[Circuit, Params]]]:
  return globals()[f'solver_{name}'](args, ctx)


def GroundStateEigensolver_solve(args, solver:MinimumEigensolver, ctx:Context) -> Union[float, Tuple[float, Tuple[Circuit, Params]]]:
  from qiskit_nature.second_q.problems.eigenstate_result import EigenstateResult
  from qiskit_nature.second_q.algorithms.ground_state_solvers import GroundStateEigensolver

  assert GroundStateEigensolver.solve
  solver_res: Result = solver.compute_minimum_eigenvalue(ctx.ham)
  eigen_res = EigenstateResult.from_result(solver_res)
  if isinstance(ctx.mol, Problem):    # PySCFDriver generated
    res = Problem.interpret(ctx.mol, eigen_res)
    if args.disp: print(res)
    gs_ene = res.total_energies.item()
  else:                               # fake
    if args.disp: print(eigen_res)
    gs_ene = eigen_res.groundenergy + ctx.mol.nuclear_repulsion_energy
  if hasattr(solver_res, 'optimal_point'):
    params_opt = list(solver_res.optimal_parameters.values())
    ansatz_opt = solver_res.optimal_circuit.assign_parameters(solver_res.optimal_parameters)
    return gs_ene, (ansatz_opt, params_opt)
  else:
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
  if False:
    # https://qiskit.org/documentation/stable/0.26/tutorials/noise/3_measurement_error_mitigation.html
    noise_model = load_noise_file(args.N) if args.N else None
    qi_noise_model_qasm = QuantumInstance(
      backend=estimator,
      shots=args.shots,
      noise_model=noise_model, 
      measurement_error_mitigation_cls=CompleteMeasFitter,
      measurement_error_mitigation_shots=args.shots_mit,
      seed_simulator=args.seed,
      seed_transpiler=args.seed,
    )
  solver.initial_point = init
  return GroundStateEigensolver_solve(args, solver, ctx)


@timer
def solver_adavqe(args, ctx:Context) -> Tuple[float, Circuit]:
  estimator = get_estimator(args)
  ansatz, init = get_ansatz(args, ctx)
  optimizer = get_optimizer(args, ansatz)
  vqe = VQE(estimator, ansatz, optimizer, callback=partial(optim_callback, args))
  solver = AdaptVQE(vqe)
  solver.initial_point = init
  return GroundStateEigensolver_solve(args, solver, ctx)


@timer
def solver_svqe(args, ctx:Context) -> Tuple[float, Circuit]:
  sampler = get_sampler(args)
  ansatz, init = get_ansatz(args, ctx)
  optimizer = get_optimizer(args, ansatz)
  solver = SamplingVQE(sampler, ansatz, optimizer, callback=partial(optim_callback, args))
  solver.initial_point = init
  return GroundStateEigensolver_solve(args, solver, ctx)


@timer
def solver_qaoa(args, ctx:Context) -> Tuple[float, Circuit]:
  sampler = get_sampler(args)
  optimizer = get_optimizer(args, None)
  solver = QAOA(sampler, optimizer, reps=args.reps, callback=partial(optim_callback, args))
  return GroundStateEigensolver_solve(args, solver, ctx)
