#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/08

from argparse import ArgumentParser
from pathlib import Path
from re import compile as Regex
from typing import List, Tuple
from math import pi

Lines = List[str]
Gate = Tuple[str, str]
Qubit = str

''' helpers '''

PGATES = ['rx', 'ry', 'rz']
R_LINE = Regex('(.+) (.+);')
R_PGATE = Regex('(\w+)\((.+)\)')

def _split_gate_qubit(line:str) -> Tuple[Gate, Qubit]:
  gate_param, qubit = R_LINE.findall(line)[0]
  m = R_PGATE.findall(gate_param)
  if m: gate, param = m[0]
  else: gate, param = gate_param, None
  return (gate, param), qubit

EPS = 1e-6
PHI_ZERO = '0'
N_DIV = 32
PI_TABLE = { pi: 'pi' }
PI_TABLE.update({  pi / n:  f'pi/{n}' for n in range(2, N_DIV+1)   })
PI_TABLE.update({ -pi / n: f'-pi/{n}' for n in range(N_DIV, 1, -1) })

def _try_rnd_to_pi(phi:float) -> str:
  if abs(phi) < EPS: return PHI_ZERO

  val = phi
  for k, v in PI_TABLE.items():
    if abs(k - phi) < EPS:
      val = v
      break
  return val

def _norm_rot_angle(phi:float) -> float:
  phi %= 2 * pi                 # vrng in (-2*pi, 2*pi)
  if phi > +pi: phi -= 2 * pi   # (pi, 2*pi) -> # (-pi, 0)
  if phi < -pi: phi += 2 * pi   # (-2*pi, -pi) -> # (0, pi)
  return phi                    # vrng in (-pi, pi)


''' processors '''

def sanitize_rot_gate_angle(lines:Lines) -> Lines:
  ''' R(phi) => R(phi') '''

  ret = []

  for this in lines:
    (g0, p0), q0 = _split_gate_qubit(this)
    if g0 in PGATES:
      phi: float = eval(p0)
      phi = _norm_rot_angle(phi)
      phi_str: str = _try_rnd_to_pi(phi)
      if phi_str != PHI_ZERO:
        ret.append(f'{g0}({phi_str}) {q0};')
    else:
      ret.append(this)

  return ret

def fold_rot_gate(lines:Lines) -> Lines:
  ''' R(psi)-R(phi) => R(psi+phi) '''

  if len(lines) < 2: return lines

  ret = [lines[0]]

  for this in lines[1:]:
    (g0, p0), q0 = _split_gate_qubit(ret[-1])
    (g1, p1), q1 = _split_gate_qubit(this)
    if q0 == q1 and g0 == g1 and g0 in PGATES:
      phi: float = eval(p0) + eval(p1)
      phi = _norm_rot_angle(phi)
      phi_str: str = _try_rnd_to_pi(phi)
      ret.pop(-1)
      if phi_str != PHI_ZERO:
        ret.append(f'{g0}({phi_str}) {q0};')
    else:
      ret.append(this)

  return ret

def fold_sx_rz_sx_rz(lines:Lines) -> Lines:
  ''' SX-RZ(pi)-SX-RZ(pi) => -jI => I'''

  if len(lines) < 4: return lines

  ret = [lines[0], lines[1], lines[2]]
  for this in lines[3:]:
    (g0, p0), q0 = _split_gate_qubit(ret[-3])
    (g1, p1), q1 = _split_gate_qubit(ret[-2])
    (g2, p2), q2 = _split_gate_qubit(ret[-1])
    (g3, p3), q3 = _split_gate_qubit(this)

    if (q0 == q1 == q2 == q3) and (g0 == g2 == 'sx') and (g1 == g3 == 'rz') and (p1 == p3 == 'pi'):
      ret.pop(-1)
      ret.pop(-1)
      ret.pop(-1)
    else:
      ret.append(this)

  return ret


''' entry '''

def optimize(qasm:str) -> str:
  def _split(qsam:str) -> Tuple[Lines, Lines]:
    lines = qsam.strip().split('\n')
    return lines[:2], lines[2:]
  def _combine(head:Lines, body:Lines) -> str:
    return '\n'.join(head + body)

  head, body = _split(qasm)
  len_raw = len(body)
  body = sanitize_rot_gate_angle(body)
  body = fold_rot_gate(body)
  body = fold_sx_rz_sx_rz(body)
  len_opt = len(body)
  print(f'>> qasm_optimize: {len_raw} => {len_opt}')
  return _combine(head, body)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('exp_dp', type=Path, help='path to log folder')
  args = parser.parse_args()

  in_fp = Path(args.exp_dp) / 'ansatz_t.raw.qsam'
  with open(in_fp, 'r', encoding='utf-8') as fh:
    qasm = fh.read()

  qasm_opt = optimize(qasm)

  out_fp = in_fp.with_stem(in_fp.stem[:-len('.raw')])
  with open(out_fp, 'w', encoding='utf-8') as fh:
    fh.write(qasm_opt)
