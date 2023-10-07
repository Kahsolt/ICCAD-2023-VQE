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
    mes_gs, (ansatz, params) = run_solver(args, args.X, ctx)
    print(f'>> mes_gs: {mes_gs:}')

    err = abs(mes_gs - ref_gs)
    print(f'>> Error: {err}')
    err_rate = err / abs(ref_gs)
    print(f'>> Error Rate: {err_rate:%}')

    if args.skip_pulse:
      mes_gs_t, ansatz_t, mapping, duration = None, None, None, None
      err_t, err_rate_t = None, None
    else:
      mes_gs_t, ansatz_t, mapping, duration = run_noisy_eval(args, ctx, ansatz)
      print(f'>> Duration: {duration}')
      err_t = abs(mes_gs_t - ref_gs)
      print(f'>> Error (qsam): {err_t}')
      err_rate_t = err_t / abs(ref_gs)
      print(f'>> Error Rate (qsam): {err_rate_t:%}')

  dt_end = datetime.now()

  name: str = args.name or exp_name(args)
  log_dp = LOG_PATH / name
  log_dp.mkdir(exist_ok=True)

  if 'qsam':
    with open(log_dp / 'ansatz.qsam', 'w', encoding='utf-8') as fh:
      fh.write(ansatz.qasm())
    if ansatz_t is not None:
      with open(log_dp / 'ansatz_t.qsam', 'w', encoding='utf-8') as fh:
        fh.write(ansatz_t.qasm())
      save_json(mapping, log_dp / 'ansatz_t.layout')

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
        'err_t': err_t,
        'err_rate_t': err_rate_t,
        'duration': duration,
      },
      'ansatz': {
        'n_qubit': ansatz.num_qubits,
        'n_ancilla': ansatz.num_ancillas,
        'n_cbit': ansatz.num_clbits,
        'size': ansatz.size(),
        'depth': ansatz.depth(),
        'width': ansatz.width(),
        'params': params,
      },
    }
    if ansatz_t is not None:
      ansatz_t: Circuit
      data['transpiled_ansatz'] = {
        'n_qubit': ansatz_t.num_qubits,
        'n_ancilla': ansatz_t.num_ancillas,
        'n_cbit': ansatz_t.num_clbits,
        'size': ansatz_t.size(),
        'depth': ansatz_t.depth(),
        'width': ansatz_t.width(),
      }

    save_json(data, log_dp / 'log.json')


if __name__ == '__main__':
  args = get_args()
  print(vars(args))
  run(args)
