@ECHO OFF

:baseline
python run.py -H txt

:less_shots
python run.py -H txt --shots 3000
python run.py -H txt --shots 1000
python run.py -H txt --shots 300
python run.py -H txt --shots 100
python run.py -H txt --shots 10

:ham_trim
python run.py -H txt -T 100 --thresh 0.001
python run.py -H txt -T 100 --thresh 0.01
python run.py -H txt -T 100 --thresh 0.1
python run.py -H txt -T 100 --thresh 0.3

python run.py -H txt -T 1000 --thresh 0.3

:solver
REM This does NOT run, due to Mem resource limitation :(
python run.py -H txt -X adavqe
REM These do NOT run, due to problem constrains mismatch :(
python run.py -H txt -X svqe
python run.py -H txt -X qaoa

:optim
python run.py -H txt -A slsqp
python run.py -H txt -A pbfgs
python run.py -H txt -A lbfgsb
python run.py -H txt -A spsa
python run.py -H txt -A qnspsa

:ansatz
python run.py -H txt -A uccs
python run.py -H txt -A uccd
python run.py -H txt -A uccsd
python run.py -H txt -A uccst
python run.py -H txt -A uccdt
python run.py -H txt -A uccsdt

python run.py -H txt -A puccd
python run.py -H txt -A succd

python run.py -H txt -A chccs
python run.py -H txt -A chccd
python run.py -H txt -A chccsd

:init
python run.py -H txt -I hf
python run.py -H txt -I mp2
python run.py -H txt -I none
python run.py -H txt -I zeros
python run.py -H txt -I randu
python run.py -H txt -I randn
