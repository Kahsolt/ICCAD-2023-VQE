# converted from `QC-Contest-Demo/NoiseModel_and_SystemModel.ipynb`

import pickle
import warnings ; warnings.filterwarnings('ignore')

from qiskit import pulse
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.random import random_circuit
from qiskit.utils import algorithm_globals
from qiskit.providers.fake_provider import FakeMontreal
from qiskit.providers.aer.noise import NoiseModel
from qiskit_aer.primitives import Estimator

seeds = 170
algorithm_globals.random_seed = seeds
seed_transpiler = seeds
shot = 6000     # NOTE: fixed as contest required


# For each noise model, we could design & submit only one optimal circuit (with trained parameters)
with open('QC-Contest-Demo/NoiseModel/fakekolkata.pkl', 'rb') as file:
  noise_model_data = pickle.load(file)
noise_model = NoiseModel().from_dict(noise_model_data)
print('noise_model:', noise_model)

circuit = random_circuit(2, 2, seed=0).decompose(reps=1)
circuit.draw()
system_model = FakeMontreal()
transpiled_circuit = transpile(circuit, backend=system_model)

observable = SparsePauliOp('IIIIIIIIIIIIIIIIIIIIIIIIIXZ')
print('observable:', observable.paulis)

# Here you can do
# - Pauli grouping
# - Error mitigation
# - Transpiling optimization

estimator = Estimator(
  backend_options = {
    'method': 'statevector',
    'device': 'CPU',
    'noise_model': noise_model,
  },
  run_options = {
    'shots': shot,
    'seed': seeds,
  },
  skip_transpilation=True,
)

job = estimator.run(transpiled_circuit, observable)
result = job.result()
print('result:', result)

with pulse.build(system_model) as prog:
  with pulse.transpiler_settings(optimization_level=0):
    pulse.call(transpiled_circuit)
print('duration:', prog.duration)

# This is the answer for submission
qasm = transpiled_circuit.qasm()
print('[QASM]')
print(qasm)
