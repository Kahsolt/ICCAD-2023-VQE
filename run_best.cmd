:noiseless_train

REM Train an ansatz without noise using VQE
REM then apply tricks like error mitigation in noisy runtime

SET NAME=noiseless_cobyla
python run.py -H txt -A chcs -O cobyla -T 500 --tol 1e-6 --shots 500 --name %NAME%
python run_eval.py log\%NAME%
python run_eval.py log\%NAME% -N cairo
python run_eval.py log\%NAME% -N kolkata
python run_eval.py log\%NAME% -N montreal

SET NAME=noiseless_spsa
python run.py -H txt -A chcs -O spsa -T 500 --tol 1e-6 --shots 500 --name %NAME%
python run_eval.py log\%NAME%
python run_eval.py log\%NAME% -N cairo
python run_eval.py log\%NAME% -N kolkata
python run_eval.py log\%NAME% -N montreal


:noisy_train

REM Is this possible? Alike adversarial training :(

SET NAME=noisy_cairo
python run.py -H txt -N cairo -A chcs -O cobyla -T 100 --tol 1e-6 --shots 500 --name %NAME%
python run_eval.py log\%NAME%
python run_eval.py log\%NAME% -N cairo
python run_eval.py log\%NAME% -N kolkata
python run_eval.py log\%NAME% -N montreal

SET NAME=noisy_kolkata
python run.py -H txt -N kolkata -A chcs -O cobyla -T 100 --tol 1e-6 --shots 500 --name %NAME%
python run_eval.py log\%NAME%
python run_eval.py log\%NAME% -N cairo
python run_eval.py log\%NAME% -N kolkata
python run_eval.py log\%NAME% -N montreal

SET NAME=noisy_montreal
python run.py -H txt -N montreal -A chcs -O cobyla -T 100 --tol 1e-6 --shots 500 --name %NAME%
python run_eval.py log\%NAME%
python run_eval.py log\%NAME% -N cairo
python run_eval.py log\%NAME% -N kolkata
python run_eval.py log\%NAME% -N montreal


REM This takes even much more time :(

SET NAME=noisy_cairo_spsa
python run.py -H txt -N cairo -A chcs -O spsa -T 100 --tol 1e-6 --shots 500 --name %NAME%
python run_eval.py log\%NAME%
python run_eval.py log\%NAME% -N cairo
python run_eval.py log\%NAME% -N kolkata
python run_eval.py log\%NAME% -N montreal

SET NAME=noisy_kolkata_spsa
python run.py -H txt -N kolkata -A chcs -O spsa -T 100 --tol 1e-6 --shots 500 --name %NAME%
python run_eval.py log\%NAME%
python run_eval.py log\%NAME% -N cairo
python run_eval.py log\%NAME% -N kolkata
python run_eval.py log\%NAME% -N montreal

SET NAME=noisy_montreal_spsa
python run.py -H txt -N montreal -A chcs -O spsa -T 100 --tol 1e-6 --shots 500 --name %NAME%
python run_eval.py log\%NAME%
python run_eval.py log\%NAME% -N cairo
python run_eval.py log\%NAME% -N kolkata
python run_eval.py log\%NAME% -N montreal
