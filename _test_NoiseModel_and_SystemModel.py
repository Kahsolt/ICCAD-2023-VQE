# converted from `QC-Contest-Demo/NoiseModel_and_SystemModel.ipynb`

import pickle
from pathlib import Path
from argparse import ArgumentParser

from qiskit import pulse, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.random import random_circuit
from qiskit.providers.fake_provider import FakeMontreal
from qiskit.utils import algorithm_globals
from qiskit_aer.primitives import Estimator
from qiskit_aer.noise import NoiseModel

BASE_PATH = Path(__file__).parent.absolute()
NOISE_MODEL_PATH = BASE_PATH / 'QC-Contest-Demo' / 'NoiseModel'


def run(args):
  # For each noise model, we could design & submit only one optimal circuit (with trained parameters)
  with open(NOISE_MODEL_PATH / f'fake{args.M}.pkl', 'rb') as fh:
    noise_model_data = pickle.load(fh)
  noise_model = NoiseModel().from_dict(noise_model_data)
  print(noise_model)
  print()

  circuit = random_circuit(2, 2, seed=args.seed).decompose(reps=1)
  out = circuit.draw(output='text')
  print(out)
  print()

  # transpile := simplify + optimize circuit towards known hardware backend
  # https://qiskit.org/documentation/apidoc/transpiler.html
  system_model = FakeMontreal()
  transpiled_circuit = transpile(circuit, backend=system_model)
  out = transpiled_circuit.draw(output='text')
  print(out)
  print()

  observable = SparsePauliOp('IIIIIIIIIIIIIIIIIIIIIIIIIXZ')
  print('observable:', observable.paulis)
  print()

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
      'shots': 6000,
      'seed': args.seed,
    },
    skip_transpilation=True,
  )

  job = estimator.run(transpiled_circuit, observable)
  result = job.result()
  print('result:', result)
  print()

  with pulse.build(system_model) as prog:
    with pulse.transpiler_settings(optimization_level=0):
      pulse.call(transpiled_circuit)
  print('duration:', prog.duration)
  print()
  
  # This is the answer for submission
  qasm = transpiled_circuit.qasm()
  print(qasm)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', default='montreal', choices=['cairo', 'kolkata', 'montreal'], help='noise model')
  parser.add_argument('--seed', default=170, type=int, help='rand seed')
  args = parser.parse_args()

  algorithm_globals.random_seed = args.seed

  run(args)
