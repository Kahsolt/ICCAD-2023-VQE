REM pretrain *.qasm files for final submission

python run.py -H txt -A chcs -I noise -O cobyla -T 100 --tol 1e-8 --shots 500 --name final_cairo
python run.py -H txt -A chcs -I noise -O cobyla -T 100 --tol 1e-8 --shots 500 --name final_kolkata
python run.py -H txt -A chcs -I noise -O cobyla -T 100 --tol 1e-8 --shots 500 --name final_montreal
