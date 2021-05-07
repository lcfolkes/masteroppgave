from pathlib import Path
home = str(Path.home())
if home == '/Users/mathiasklev':
    src_path = '/Documents/10.semester/masteroppgave/src'
    path_to_src = home + src_path
elif home == '/Users/larsfolkestad':
    src_path = '/Documents/Skole/NTNU/Master/masteroppgave/src/'
    path_to_src = home + src_path
elif home == '/home/icfolkes':
    src_path = '/storage/users/icfolkes/masteroppgave/src/'
    path_to_src = src_path

elif home == '/home/mathiadk':
    src_path = '/storage/users/mathiadk/masteroppgave/src/'
    path_to_src = src_path
