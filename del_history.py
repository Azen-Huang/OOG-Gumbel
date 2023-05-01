from pathlib import Path
import numpy as np
import pickle
import os

dir = './train_data'
def del_hitory():
    board_history_path = sorted(Path(dir + '/board').glob('*.history'))[0]
    policies_history_path = sorted(Path(dir + '/policies').glob('*.history'))[0]
    enemy_policies_history_path = sorted(Path(dir + '/auxiliary_policies').glob('*.history'))[0]
    value_history_path = sorted(Path(dir + '/value').glob('*.history'))[0]
    
    os.remove(board_history_path)
    os.remove(policies_history_path)
    os.remove(enemy_policies_history_path)
    os.remove(value_history_path)

if __name__ == '__main__':
    del_hitory()