# converted from `QC-Contest-Demo/examplecode.ipynb`

from time import time

from qiskit import pulse
from qiskit.algorithms.minimum_eigensolvers import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SLSQP
from qiskit.utils import algorithm_globals
from qiskit.providers.fake_provider import FakeMontreal
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_aer.primitives import Estimator

seeds = 170
algorithm_globals.random_seed = seeds
seed_transpiler = seeds
shot = 6000       # NOTE: fixed as contest required
iterations = 10


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
qmolecule = driver.run()

hamiltonian = qmolecule.hamiltonian
coefficients = hamiltonian.electronic_integrals
print(coefficients.alpha)
second_q_op = hamiltonian.second_q_op()

mapper = JordanWignerMapper()
converter = QubitConverter(mapper=mapper, two_qubit_reduction=False)
qubit_op = converter.convert(second_q_op)

solver = GroundStateEigensolver(
  JordanWignerMapper(),
  NumPyMinimumEigensolver(),
)

result = solver.solve(qmolecule)
print('computed_energies:', result.computed_energies)
print('nuclear_repulsion_energy:', result.nuclear_repulsion_energy)

ref_value = result.computed_energies + result.nuclear_repulsion_energy
print('ref_value:', ref_value)

ansatz = UCCSD(
  qmolecule.num_spatial_orbitals,
  qmolecule.num_particles,
  mapper,
  initial_state=HartreeFock(
    qmolecule.num_spatial_orbitals,
    qmolecule.num_particles,
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
    'shots': shot,
    'seed': seeds,
  },
  transpile_options = {
    'seed_transpiler': seed_transpiler
  }
)

vqe_solver = VQE(estimator, ansatz, SLSQP())
vqe_solver.initial_point = [0.0] * ansatz.num_parameters

start_time = time()
solver = GroundStateEigensolver(mapper, vqe_solver)
res = solver.solve(qmolecule)
end_time = time()
print('solver res:', res)
print('time cost:', end_time - start_time)

result = res.computed_energies + res.nuclear_repulsion_energy
print('solver gs:', result)
error_rate = abs(abs(ref_value - result) / ref_value * 100)
print(f'Error rate: {error_rate:%}')

system_model = FakeMontreal()
with pulse.build(system_model) as prog:
  pulse.call(ansatz)
print('duration:', prog.duration)
