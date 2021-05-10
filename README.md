# Master Thesis
This is a repository for the master thesis of Lars Folkestad and Mathias Klev optimizing staff-based relocation of
an electric car-sharing system under demand uncertainty. 

## Setup

### Gurobi
This project uses gurobi as a solver for the optimization problem. To use gurobi we need a license file and the program
itself. [This quickstart](https://www.gurobi.com/wp-content/plugins/hd_documentations/content/pdf/quickstart_mac_8.1.pdf#page=89&zoom=100,96,96)
 is a good starting point to obtain both of these.

### Anaconda environment
It is recommended to use anaconda for gurobi. [Download anaconda](https://docs.anaconda.com/anaconda/install/mac-os/) 
(I used the command-line install) and create an environment from a environment.yml file.


This whole tutorial is based on [the conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

Change directory into the root directory and run the following command:
```
conda env create --prefix ./env -f environment.yml
```
This command creates a conda environment at ./env in your current directory with all the required packages for the project.
PS: To remove the awful looking path on the left when activating your environment run `conda config --set env_prompt '({name})'`
when the environment is active. You need to restart your terminal to see this change. 

####Adding new packages to conda environment
1. Search for package using `conda search <package>` to check for availability
2. Add package name to environment.yml under dependencies
3. run the following command to update your conda environment with the new dependencies:
```
conda env update --prefix ./env --file environment.yml  --prune
```
To remove packages just remove it from the packages.yml and run the same command. 
This is a nice command to make a alias for.

## Solstorm

### Login
```
ssh solstorm-login.iot.ntnu.no -l <username>
```

### Pull from github in Solstorm login node or screen
```
cd /storage/users/<username>/masteroppgave
git pull
```

### Setup
```
cd /storage/users/<username>/masteroppgave
pip install -r requirements.txt
```


### Screen
To run from a compute node one must first connect to a *screen*.
#### Check if you have any screens
```
screen -ls
```
#### Connect to existing screen
```
screen -R <screen_name>
```

#### Create new screen
```
screen -S <screen_name>
```

#### Detach from screen
<kbd>Ctrl + a</kbd> <kbd>d</kbd>

### Connect to compute node
When connected to a screen, you must connect to a compute node before running any tests.
```
ssh compute-<rack>-<node_id>
```
#### Load modules on new compute node
The first time you access a compute node, you must load Python and Gurobi
```
module load Python/3.8.6-GCCcore-10.2.0
module load gurobi/9.1
cd $GUROBI_HOME
python setup.py build -b $HOME/.cache/gurobipy install --user
```
Navigate back to *storage* and pip install requirements (see Solstorm->Setup)

## Run from terminal
To run a file from the terminal one must first set the <code>PYTHONPATH</code>. First, navigate to the root directory.
```
cd masteroppgave/src
```
Then update <code>PYTHONPATH</code> with the following command:
```
export PYTHONPATH="$PWD"
```
To confirm that the root directory is added to <code>PYTHONPATH</code>, type:
```
echo $PYTHONPATH
```
When <code>PYTHONPATH</code> is set correctly, simply run the file. Remember to specify path relative to <code>/src</code>. E.g.:
```
python Heuristics/main.py
```
