#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/13

from utils import *
from solvers import run_solver, fxs


def exp_cfg(args) -> str:
  cfg = f'{args.X}_{args.ansatz}_{args.init}_{args.reps}_{args.O}_{args.maxiter}'
  if args.N: cfg += f'_{args.N}'
  return cfg

def exp_name(args) -> str:
  ts = datetime.now().isoformat()
  ts = ts.replace('-', '').replace(':', '').replace('T', '_')
  ts = ts.split('.')[0]
  cfg = exp_cfg(args)
  return f'{ts}@{cfg}'


def run(args):
  dt_start = datetime.now()

  if 'experiment':
    ctx = get_context(args)
  
    ref_gs = run_solver(args, args.Y, ctx)
    print(f'>> ref_gs: {ref_gs:}')
    mes_gs, ansatz = run_solver(args, args.X, ctx)
    print(f'>> mes_gs: {mes_gs:}')

    err = abs(mes_gs - ref_gs)
    print(f'>> Error: {err}')
    err_rate = err / abs(ref_gs)
    print(f'>> Error Rate: {err_rate:%}')

    if args.skip_pulse:
      schedule = None
      duration, ansatz_t = None, None
    else:
      schedule, ansatz_t = run_pulse(args, ansatz)
      duration = schedule.duration
      print(f'>> Duration: {duration}')

  dt_end = datetime.now()

  name: str = args.name or exp_name(args)
  log_dp = LOG_PATH / name
  log_dp.mkdir(exist_ok=True)

  if 'pulse' and schedule is not None:
    with open(log_dp / 'pulse.txt', 'w', encoding='utf-8') as fh:
      fh.write(str(schedule))

  if 'plot':
    plt.rcParams['figure.figsize'] = (10, 4)
    plt.plot(fxs, c='r')
    plt.ylabel('Energy')
    plt.suptitle(exp_cfg(args))
    plt.savefig(log_dp / 'energy.png', dpi=600)

  if 'log':
    data = {
      'args': vars(args),
      'cmd': ' '.join(sys.argv),
      'tm_start': str(dt_start),
      'tm_end': str(dt_end),
      'tm_cost': (dt_end - dt_start).total_seconds(),
      'metrics': {
        'ref_gs': ref_gs,
        'mes_gs': mes_gs,
        'err': err,
        'err_rate': err_rate,
        'duration': duration,
      },
      'ansatz': {
        'n_qubit': ansatz.num_qubits,
        'n_ancilla': ansatz.num_ancillas,
        'n_cbit': ansatz.num_clbits,
        'size': ansatz.size(),
        'depth': ansatz.depth(),
        'width': ansatz.width(),
      },
    }
    if ansatz_t is not None:
      data['transpiled_ansatz'] = {
        'n_qubit': ansatz_t.num_qubits,
        'n_ancilla': ansatz_t.num_ancillas,
        'n_cbit': ansatz_t.num_clbits,
        'size': ansatz_t.size(),
        'depth': ansatz_t.depth(),
        'width': ansatz_t.width(),
        #'qasm': ansatz_t.qasm(),
      }
    
    save_json(data, log_dp / 'log.json')


if __name__ == '__main__':
  parser = ArgumentParser()
  # hardware
  parser.add_argument('-S', default='Montreal', choices=['Montreal'], help='system model (27 qubits)')
  parser.add_argument('-N', default='', choices=['', 'cairo', 'kolkata', 'montreal'], help='noise model data (27 qubits)')
  parser.add_argument('--simulator', default='automatic', choices=SIMULATORS, help='qiskit_aer simulator method')
  parser.add_argument('--shots', default=6000, help='shot-based simulator resource limit')
  # ham
  parser.add_argument('-H', default='run', help=f'"run" requires PySCF, "txt" loads {HAM_FILE}; or filepath to your own ham, see fmt in `utils.load_ham_file()`')
  parser.add_argument('--thresh', default=1e-6, type=float, help='ham term trim amplitude thresh')
  # eigensolver
  parser.add_argument('-Y', default='numpy', choices=['numpy', 'const'], help='classical eigensolver (ref_val)')
  parser.add_argument('-X', default='vqe', choices=['vqe', 'adavqe', 'svqe', 'qaoa'], help='quantum eigensolver')
  # ansatz
  parser.add_argument('-A', '--ansatz', default='uccsd', choices=ANSATZS, help='ansatz model')
  parser.add_argument('-I', '--init',   default='zeros', choices=['zeros', 'randu', 'randn'], help='ansatz param init')
  parser.add_argument('--reps', default=1, help='ansatz circuit n_repeats')
  # optim
  parser.add_argument('-O', default='cobyla', choices=OPTIMZERS.keys(), help='optim method')
  parser.add_argument('-T', '--maxiter', default=10, type=int, help='optim maxiter')
  parser.add_argument('--tol', default=1e-5, type=float, help='optim tolerance')
  parser.add_argument('--disp', action='store_true', help='optim show verbose result')
  # misc
  parser.add_argument('--seed', default=170, type=int, help='rand seed')
  parser.add_argument('--skip_pulse', action='store_true', help='skip run_pulse')
  parser.add_argument('--name', help='experiment name under log folder, default to time-string')
  args = parser.parse_args()

  if args.H == 'txt': args.H = str(HAM_FILE)
  if args.H != 'run': assert Path(args.H).is_file(), f'-H {args.H!r} should be a valid file'

  seed_everything(args.seed)

  args.device = 'GPU' if 'GPU' in AerSimulator().available_devices() else 'CPU'

  print(vars(args))
  run(args)
