module load Python/3.8.6-GCCcore-10.2.0
module load gurobi/9.1
cd $GUROBI_HOME
python setup.py build -b $HOME/.cache/gurobipy install --user
cd ../../../../storage/users/$USER/masteroppgave/
pip install -r requirements.txt
cd src
export PYTHONPATH="$PWD"
printf "\n-------------------- Python setup complete --------------------\n\n"
git status
