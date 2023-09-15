@ECHO OFF

:baseline
python run.py -H txt -O cobyla -T 10

:noisy
REM These does NOT run, consuming 100% CPU, do not know why :(
python run.py -H txt -O cobyla -T 10 -N cairo
python run.py -H txt -O cobyla -T 10 -N kolkata
python run.py -H txt -O cobyla -T 10 -N montreal

:less_shots
python run.py -H txt -O cobyla -T 10 --shots 3000
python run.py -H txt -O cobyla -T 10 --shots 1000
python run.py -H txt -O cobyla -T 10 --shots 300
python run.py -H txt -O cobyla -T 10 --shots 100
python run.py -H txt -O cobyla -T 10 --shots 10

:ham_trim
python run.py -H txt -O cobyla -T 100 --thresh 0.001
python run.py -H txt -O cobyla -T 100 --thresh 0.01
python run.py -H txt -O cobyla -T 100 --thresh 0.1
python run.py -H txt -O cobyla -T 100 --thresh 0.3

python run.py -H txt -O cobyla -T 1000 --thresh 0.3
