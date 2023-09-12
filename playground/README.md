# Why is the Qiskit base code so slow & inaccurate?

    we compare with other frameworks first :(

----

ℹ The code are migrated from repo [Kahsolt/Mindquantum-Hackathon-2023-VQE](https://github.com/Kahsolt/Mindquantum-Hackathon-2023-VQE/tree/main/stage1/Q1/playground)  
⚠ The dependency packages are only available on **Linux**!  


### ChemiQ

- read doc: [https://pychemiq-tutorial.readthedocs.io/en/latest/index.html](https://pychemiq-tutorial.readthedocs.io/en/latest/index.html)
- `conda create -n chq python==3.8`
- `conda activate chq`
- `pip install pychemiq numpy`
- `python run_chq.py`

⚠ pauli terms count is correct (631), but the coeffs mismatch with the given data, why? 🤔


### Mindquantum

- read doc: [https://gitee.com/mindspore/mindquantum](https://gitee.com/mindspore/mindquantum)
- `conda create -n mq python==3.9`
- `conda activate mq`
- install mindspore
  - follow `https://www.mindspore.cn/install`
  - install combination `2.0.0 + CPU + Windows-x64 + Python 3.9 + Pip`
  - fix PIL version compatible error: `pip install Pillow==9.5.0`
  - test installation `python -c "import mindspore;mindspore.run_check()"`
- `pip install -r requirements.txt`
- `python run_mq.py`

----
by Armit
2023/09/12
