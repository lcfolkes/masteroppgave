from pathlib import Path
home = str(Path.home())
if home == '/Users/mathiasklev':
    src_path = '/Documents/10.semester/masteroppgave/src'
elif home == '/Users/larsfolkestad':
    src_path = '/Documents/Skole/NTNU/Master/masteroppgave/src/'
elif home == '/home/icfolkes':
    src_path = '/storage/users/icfolkes/masteroppgave/src/'
elif home == '/home/mathiadk':
    src_path = '/storage/users/mathiadk/masteroppgave/src/'
path_to_src = home+src_path
