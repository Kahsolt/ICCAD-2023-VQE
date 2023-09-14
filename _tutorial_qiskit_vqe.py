#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/14

# https://qiskit.org/ecosystem/algorithms/tutorials/index.html

from qiskit.quantum_info import SparsePauliOp

H2_op = SparsePauliOp.from_list([
  ('II', -1.052373245772859),
  ('IZ', +0.39793742484318045),
  ('ZI', -0.39793742484318045),
  ('ZZ', -0.01128010425623538),
  ('XX', +0.18093119978423156)
])
print(H2_op)
num_qubits = H2_op.num_qubits
print(f'num_qubits: {num_qubits}')

print('=' * 72)


''' Chapter 1: An Introduction to Algorithms in Qiskit '''

from qiskit.primitives import Estimator
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import SLSQP, SPSA
from qiskit.algorithms.minimum_eigensolvers import VQE

# analytical simulator & optimizer (for theoretical simulation)
estimator = Estimator()
ansatz = TwoLocal(num_qubits, 'ry', 'cz')
optimizer = SLSQP(maxiter=1000)
vqe = VQE(estimator, ansatz, optimizer)
result = vqe.compute_minimum_eigenvalue(H2_op)
print(result)

# shot-based simulator & optimizer (for noisy simulation)
estimator = Estimator(options={'shots': 1000})
vqe.estimator = estimator
vqe.optimizer = SPSA(maxiter=100)
result = vqe.compute_minimum_eigenvalue(H2_op)
print(result)


print('=' * 72)


''' Chapter 3: VQE with Qiskit Aer Primitives '''

# classical solver
from qiskit.opflow import PauliSumOp
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

numpy_solver = NumPyMinimumEigensolver()
result = numpy_solver.compute_minimum_eigenvalue(PauliSumOp(H2_op))
ref_value = result.eigenvalue.real
print(f'Reference value: {ref_value:.5f}')


# vqe solver
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import SPSA
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.utils import algorithm_globals

seed = 170
iterations = 125

algorithm_globals.random_seed = seed
ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
spsa = SPSA(maxiter=iterations)

counts, values = [], []
def store_intermediate_result(eval_count, parameters, mean, std):
  global counts, values
  counts.append(eval_count)
  values.append(mean)


# performance without noise
from qiskit_aer.primitives import Estimator as AerEstimator

noiseless_estimator = AerEstimator(
  run_options={'seed': seed, 'shots': 1024},
  transpile_options={'seed_transpiler': seed},
)
vqe = VQE(noiseless_estimator, ansatz, optimizer=spsa, callback=store_intermediate_result)
result0 = vqe.compute_minimum_eigenvalue(H2_op).eigenvalue.real

print(f'VQE on Aer qasm simulator (no noise): {result0:.5f}')
print(f'Delta from reference energy value is {(result0 - ref_value):.5f}')

# re-start callback variables
from copy import deepcopy
counts0, values0 = deepcopy(counts), deepcopy(values)
counts, values = [], []

# performance with noise
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeVigo

device = FakeVigo()
coupling_map = device.configuration().coupling_map
noise_model = NoiseModel.from_backend(device)
print(noise_model)

noisy_estimator = AerEstimator(
  backend_options={
    'method': 'density_matrix',
    'coupling_map': coupling_map,
    'noise_model': noise_model,
  },
  run_options={'seed': seed, 'shots': 1024},
  transpile_options={'seed_transpiler': seed},
)
vqe.estimator = noisy_estimator
result1 = vqe.compute_minimum_eigenvalue(H2_op).eigenvalue.real

print(f'VQE on Aer qasm simulator (with noise): {result1:.5f}')
print(f'Delta from reference energy value is {(result1 - ref_value):.5f}')

print('-' * 36)

# performance comparation
print(f'Reference value: {ref_value:.5f}')
print(f'VQE on Aer qasm simulator (no noise): {result0:.5f}')
print(f'VQE on Aer qasm simulator (with noise): {result1:.5f}')

# plot
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12, 4)
plt.plot(counts0, values0, c='b', label='w/o noise') ; plt.xlabel('Eval count') ; plt.ylabel('Energy')
plt.plot(counts,  values,  c='r', label='w/  noise') ; plt.xlabel('Eval count') ; plt.ylabel('Energy')
plt.suptitle('H2 vqe-spsa')
plt.legend()
plt.show()
