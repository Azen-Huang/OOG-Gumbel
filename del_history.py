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
    print('delet board: ', board_history_path)
    print('delet policy: ', policies_history_path)
    print('delet enemy policy: ', enemy_policies_history_path)
    print('delet value: ', value_history_path)
    print('')

def del_all():
    sz = Path(dir + '/board').glob('*.history')
    for _ in sz:
        print(_)
        del_hitory()

if __name__ == '__main__':
    #del_all()
    del_hitory()