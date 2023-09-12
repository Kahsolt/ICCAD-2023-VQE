# converted from `QC-Contest-Demo/examplecode.ipynb`

from pathlib import Path
from time import time
from argparse import ArgumentParser

from qiskit import pulse
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.algorithms.minimum_eigensolvers import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SLSQP, L_BFGS_B, P_BFGS, COBYLA, CG, SLSQP, ADAM
from qiskit.providers.fake_provider import FakeMontreal
from qiskit_aer.primitives import Estimator
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.drivers.pyscfd.pyscfdriver import ElectronicStructureProblem
from qiskit_nature.second_q.operators.fermionic_op import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

from utils import *

Molecular = ElectronicStructureProblem
Mapper = JordanWignerMapper


def get_mol(args) -> Tuple[Molecular, Mapper]:
  ultra_simplified_ala_string = '''
  O 0.0 0.0 0.0
  H 0.45 -0.1525 -0.8454
  '''
  driver = PySCFDriver(
    atom=ultra_simplified_ala_string.strip(),
    basis='sto3g',
    charge=1,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
  )
  mol = driver.run()
  ham = mol.hamiltonian

  if 'debug':
    mol.hamiltonian
    mol.nuclear_repulsion_energy
    mol.num_particles
    mol.num_alpha
    mol.num_beta
    mol.num_spin_orbitals
    mol.num_spatial_orbitals
    mol.orbital_energies
    mol.orbital_energies_b
    mol.orbital_occupations
    mol.orbital_occupations_b

    ham.constants
    
    coeffs = ham.electronic_integrals
    coeffs.alpha
    coeffs.beta
    coeffs.alpha_beta
    coeffs.beta_alpha
    coeffs.one_body
    coeffs.two_body
    coeffs.register_length    # := .alpha.register_length = 6

  second_q_op: FermionicOp = ham.second_q_op()  # fermi
  second_q_op = second_q_op.normal_order()
  print('len(second_q_op):', len(second_q_op))  # 1860 / 630 (if apply normal_order)

  if 'debug':
    second_q_op.num_spin_orbitals
    second_q_op.register_length   # := num_spin_orbitals

  mapper = JordanWignerMapper()
  if 'new API':
    qubit_op: PauliSumOp = mapper.map(second_q_op)
  else:
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=True, sort_operators=True)
    qubit_op: PauliSumOp = converter.convert(second_q_op)   # pauli
  print('len(qubit_op):', len(qubit_op))    # 631
  
  if 'debug':
    pauli_op = qubit_op.primitive
    pauli_op.paulis           # like 'IIIIXZX...'
    pauli_op.coeffs           # like 0.05
    pauli_op.size             # 631
    pauli_op.parameters       # {}
    pauli_op.dim              # (4096, 4096)
    pauli_op.is_unitary       # False
    pauli_op.simplify         # check
    pauli_op.group_commuting  # grouping terms
    pauli_op.chop             # trim small term
    pauli_op.to_list
    pauli_op.to_matrix
    pauli_op.to_operator
    qubit_op.grouping_type    # 'None'
    qubit_op.num_qubits       # 12
    qubit_op.coeffs
    qubit_op.is_hermitian     # True

    fp = Path(__file__).with_suffix('.ham')
    if not fp.exists():
      with open(fp, 'w', encoding='utf-8') as fh:
        for coeff, pauli in zip(pauli_op.coeffs, pauli_op.paulis):
          fh.write(f'{coeff.real} * {pauli}\n')

  return mol, mapper


def solver_numpy(args, mol:Molecular, mapper:Mapper) -> float:
  solver = GroundStateEigensolver(
    mapper,
    NumPyMinimumEigensolver(),
  )

  s = time()
  res = solver.solve(mol)
  t = time()
  print('time cost:', t - s)
  print(res)

  res.eigenstates
  res.eigenvalues               # [-78.75252123]
  res.hartree_fock_energy       # -73.89066949686494
  res.nuclear_repulsion_energy  # 4.36537496654537
  res.groundenergy              # -78.75252123473416
  res.total_energies            # [-74.38714627]

  numpy_gs = res.groundenergy + res.nuclear_repulsion_energy
  print('numpy_gs:', numpy_gs)  # -74.38714627
  return numpy_gs


def solver_vqe(args, mol:Molecular, mapper:Mapper, ref_value:float) -> float:
  ansatz = UCCSD(
    mol.num_spatial_orbitals, # 6
    mol.num_particles,        # (4, 4)
    mapper,
    initial_state=HartreeFock(
      mol.num_spatial_orbitals,
      mol.num_particles,
      mapper,
    ),
  )
  estimator = Estimator(
    backend_options = {
      'method': 'statevector',
      'device': 'CPU',
      #'noise_model': noise_model,
    },
    run_options = {
      'shots': shots,
      'seed': args.seed,
    },
    transpile_options = {
      'seed_transpiler': args.seed,
    }
  )
  OPTIMZERS = {
    'slsqp':  lambda: SLSQP(maxiter=args.T, ftol=args.tol, tol=args.tol, disp=True),
    'lbfgsb': lambda: L_BFGS_B(maxiter=args.T, maxfun=args.T, ftol=args.tol),
    'pbfgs':  lambda: P_BFGS(maxfun=args.T, ftol=args.tol),
    'cobyla': lambda: COBYLA(maxiter=args.T, tol=args.tol, disp=True),
    'cg':     lambda: CG(maxiter=args.T, gtol=args.tol, tol=args.tol, disp=True),
    'adam':   lambda: ADAM(maxiter=args.T, tol=args.tol),
  }
  optimizer = OPTIMZERS[args.O]()
  vqe_solver = VQE(estimator, ansatz, optimizer)
  vqe_solver.initial_point = [0.0] * ansatz.num_parameters

  print('>> vqe solver start...')

  s = time()
  solver = GroundStateEigensolver(mapper, vqe_solver)
  res = solver.solve(mol)
  t = time()
  print('time cost:', t - s)
  print(res)

  vqe_gs = res.groundenergy + res.nuclear_repulsion_energy
  print('vqe_gs:', vqe_gs)
  err = abs(ref_value - vqe_gs) / abs(ref_value)
  print(f'Error rate: {err:%}')

  system_model = FakeMontreal()
  with pulse.build(system_model) as prog:
    pulse.call(ansatz)
  print('duration:', prog.duration)

  return vqe_gs


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', default='montreal', choices=['cairo', 'kolkata', 'montreal'], help='noise model')
  parser.add_argument('-O', default='slsqp', choices=['slsqp', 'lbfgsb', 'pbfgs', 'cobyla', 'cg', 'adam'], help='optim meth')
  parser.add_argument('-T', default=100, type=int, help='optim max_iter')
  parser.add_argument('--tol',  default=1e-4, type=float, help='optim tolerance')
  parser.add_argument('--seed', default=170,  type=int, help='rand seed')
  args = parser.parse_args()

  seed_everything(args.seed)

  mol, mapper = get_mol(args)
  numpy_gs = solver_numpy(args, mol, mapper)
  vqe_gs = solver_vqe(args, mol, mapper, numpy_gs)
