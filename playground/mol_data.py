# collect from `QC-Contest-Demo/examplecode.ipynb`

def get_mol_data(fmt:str='mq'):
  assert fmt in ['chq', 'mq']

  # -OH
  geo = [
    ('O', [0.0, 0.0, 0.0 ]),
    ('H', [0.45, -0.1525, -0.8454]),
  ]
  if fmt == 'chq':
    for i, v in enumerate(geo):
      geo[i] = ' '.join([str(e) for e in [v[0]] + v[1]])

  basis  = 'sto-3g'   # 基
  spin   = 0          # 自旋
  charge = 1          # 电荷

  return geo, basis, spin, charge
