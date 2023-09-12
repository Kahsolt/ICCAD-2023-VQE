@ECHO OFF

REM ChemiQ
python run_chq.py -T UCCSD
python run_chq.py -T UCCS
python run_chq.py -T UCCD


REM Mindquantum
python run_mq.py --run_all
