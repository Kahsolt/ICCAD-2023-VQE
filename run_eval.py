#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/20

from utils import *
from solvers import solver_const


def eval(args) -> Tuple[float, int]:
  ctx = get_context(args)
  transpiled_circuit = Circuit.from_qasm_file(args.qsam_fp)
  mapping: QubitMapping = load_json(args.layout_fp)
  mapping = {int(k): v for k, v in mapping.items()}

  ref_val = solver_const(None, None)
  print(f'>> ref_val: {ref_val}')
  ene_gs = run_transpiled_circuit(args, ctx, transpiled_circuit, mapping)
  print(f'>> run qsam result: {ene_gs}, error rate: {abs(abs(ene_gs - ref_val) / ref_val):%}')

  system_model = get_backend(args)
  duration = run_pulse(system_model, transpiled_circuit)
  print('>> run pulse duration:', duration)

  return ene_gs, duration


if __name__ == '__main__':
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
  args = parser.parse_args()

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

  print(vars(args))
  eval(args)
