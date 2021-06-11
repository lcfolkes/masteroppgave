# Master Thesis
This is a repository for the master thesis of Lars Folkestad and Mathias Klev optimizing staff-based relocation of
an electric car-sharing system under demand uncertainty. The project consists of a Gurobi model and an Adaptive Large
Neighborhood Search (ALNS) heuristic for the Stochastic Electric Car-Sharing Problem (SE-CReP).

## Project Structure
The main components of the project are the instance generator, gurobi and heuristic files.

### Instance Generator 
In the instance generator directory, the world class and its components (nodes, employees, car-moves etc.) are defined.
The instance generator file populates a world object with real data from VyBil and the Google Maps DistanceMatrix API,
scaled to the instance defined by a config file in the InstanceConfigs directory. The result is a .yaml file and .pkl file
describing an instance to be read by Gurobi and the Heuristic, respectively.

### Gurobi 
In the Gurobi directory, the file defining the Gurobi model is defined. Another file is used to run the model. 

### Heuristics
In the Heuristics directory, all classes related to the ALNS heuristic are defined. This includes, the construction heuristic,
the destroy and repair heuristics and the local search operators. The objective function and the feasibility checker are also
found here. The alns.py contains the ALNS class, and the alns algorithm can be run from here.

## Setup

### Gurobi
This project uses gurobi as a solver for the optimization problem. To use gurobi we need a license file and the program
itself. [This quickstart](https://www.gurobi.com/wp-content/plugins/hd_documentations/content/pdf/quickstart_mac_8.1.pdf#page=89&zoom=100,96,96)
 is a good starting point to obtain both of these.

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
Navigate to repository and run shell script
```
cd /storage/users/<username>/masteroppgave
source setup.sh
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

## Run from terminal
Navigate to the root directory and run
```
cd masteroppgave/src
python Testing/main.py
```
