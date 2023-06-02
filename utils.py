import numpy as np
from pathlib import Path
import numpy as np
import pickle
import os
import time
from tqdm import tqdm, trange
import shutil
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

def rotate(A):
    B = np.copy(A)
    B[:] = [[row[i] for row in B[::-1]] for i in range(len(B))]
    return B

def mirror(A):
    B = np.copy(A)
    return np.flip(B, axis = 1)

def expand_board_rotated_data(xs):
    mirror_xs = []
    for board_2 in xs:
        mirror_board_2 = []
        for board in board_2:
            mirror_board = mirror(board)
            mirror_board_2.append(mirror_board)
        mirror_xs.append(mirror_board_2)
    mirror_xs = np.array(mirror_xs)

    rotate_90 = []
    for board_2 in xs:
        rotate_board_2 = []
        for board in board_2:
            rotate_board = rotate(board)
            rotate_board_2.append(rotate_board)
        rotate_90.append(rotate_board_2)
    rotate_90 = np.array(rotate_90)

    rotate_180 = []
    for board_2 in rotate_90:
        rotate_board_2 = []
        for board in board_2:
            rotate_board = rotate(board)
            rotate_board_2.append(rotate_board)
        rotate_180.append(rotate_board_2)
    rotate_180 = np.array(rotate_180)

    rotate_270 = []
    for board_2 in rotate_180:
        rotate_board_2 = []
        for board in board_2:
            rotate_board = rotate(board)
            rotate_board_2.append(rotate_board)
        rotate_270.append(rotate_board_2)
    rotate_270 = np.array(rotate_270)
    
    mirror_rotate_90 = []
    for board_2 in mirror_xs:
        rotate_board_2 = []
        for board in board_2:
            rotate_board = rotate(board)
            rotate_board_2.append(rotate_board)
        mirror_rotate_90.append(rotate_board_2)
    mirror_rotate_90 = np.array(mirror_rotate_90)

    mirror_rotate_180 = []
    for board_2 in mirror_rotate_90:
        rotate_board_2 = []
        for board in board_2:
            rotate_board = rotate(board)
            rotate_board_2.append(rotate_board)
        mirror_rotate_180.append(rotate_board_2)
    mirror_rotate_180 = np.array(mirror_rotate_180)

    mirror_rotate_270 = []
    for board_2 in mirror_rotate_180:
        rotate_board_2 = []
        for board in board_2:
            rotate_board = rotate(board)
            rotate_board_2.append(rotate_board)
        mirror_rotate_270.append(rotate_board_2)
    mirror_rotate_270 = np.array(mirror_rotate_270)
    
    xs = np.concatenate([xs, rotate_90])
    xs = np.concatenate([xs, rotate_180])
    xs = np.concatenate([xs, rotate_270])
    xs = np.concatenate([xs, mirror_xs])
    xs = np.concatenate([xs, mirror_rotate_90])
    xs = np.concatenate([xs, mirror_rotate_180])
    xs = np.concatenate([xs, mirror_rotate_270])
    xs = xs.transpose(0,2,3,1).astype('float32')
    return xs

def expand_policy_rotated_data(policies):  
    mirror_policies = []
    for board in policies:
        mirror_board = mirror(board)
        mirror_policies.append(mirror_board)
    mirror_policies = np.array(mirror_policies)
    #print(mirror_policies[9], '\n')
    
    rotate_90 = []
    for board in policies:
        rotate_board = rotate(board)
        rotate_90.append(rotate_board)
    rotate_90 = np.array(rotate_90)
    #print(rotate_90[9],'\n')

    rotate_180 = []
    for board in rotate_90:
        rotate_board = rotate(board)
        rotate_180.append(rotate_board)
    rotate_180 = np.array(rotate_180)
    #print(rotate_180[9],'\n')

    rotate_270 = []
    for board in rotate_180:
        rotate_board = rotate(board)
        rotate_270.append(rotate_board)
    rotate_270 = np.array(rotate_270)
    #print(rotate_270[9],'\n')
    
    mirror_rotate_90 = []
    for board in mirror_policies:
        rotate_board = rotate(board)
        mirror_rotate_90.append(rotate_board)
    mirror_rotate_90 = np.array(mirror_rotate_90)
    #print(mirror_rotate_90[9],'\n')

    mirror_rotate_180 = []
    for board in mirror_rotate_90:
        rotate_board = rotate(board)
        mirror_rotate_180.append(rotate_board)
    mirror_rotate_180 = np.array(mirror_rotate_180)
    #print(mirror_rotate_180[9],'\n')

    mirror_rotate_270 = []
    for board in mirror_rotate_180:
        rotate_board = rotate(board)
        mirror_rotate_270.append(rotate_board)
    mirror_rotate_270 = np.array(mirror_rotate_270)
    #print(mirror_rotate_270[9],'\n')
    
    policies = np.concatenate([policies, rotate_90])
    policies = np.concatenate([policies, rotate_180])
    policies = np.concatenate([policies, rotate_270])
    policies = np.concatenate([policies, mirror_policies])
    policies = np.concatenate([policies, mirror_rotate_90])
    policies = np.concatenate([policies, mirror_rotate_180])
    policies = np.concatenate([policies, mirror_rotate_270])
    policies = policies.reshape(len(policies), 15 * 15).astype('float32')
    return policies

def expand_value_rotated_data(value):
    copy_value = np.copy(value)
    value = np.concatenate([value, copy_value])
    value = np.concatenate([value, copy_value])
    value = np.concatenate([value, copy_value])
    value = np.concatenate([value, copy_value])
    value = np.concatenate([value, copy_value])
    value = np.concatenate([value, copy_value])
    value = np.concatenate([value, copy_value])
    value = value.astype('float32')
    return value

def load_data(path, SIZE, file_type):
    if(SIZE == 0):
        print(path + ' have not file.')
        return

    if (file_type == '*.history'):
        data = []
        for i in range(0, SIZE):
            history_path = sorted(Path(path).glob(file_type))[i]
            #print(history_path)
            with history_path.open(mode = 'r') as f:
                vec = f.readlines()
                f.close()
            
            output_arr = []
            for v in vec:
                v = v.split(", ")
                output_arr.append(v[:-1])
            
            if (path[-5:] == 'value'):
                data += output_arr[0]
            else:
                data += output_arr
        return data
    elif(file_type == '*.npy'):
        data = np.load(sorted(Path(path).glob(file_type))[0])
        for i in range(1, SIZE):
            history_path = sorted(Path(path).glob(file_type))[i]
            data = np.concatenate([data, np.load(history_path)])
        return data

def del_history(dir, count = 1, file_type = '*.history'):
    print('Delete',dir, 'History data. count:', count)
    for _ in tqdm(range(count)):
        board_history_path = sorted(Path(dir + '/board').glob(file_type))[0]
        os.remove(board_history_path)
        policies_history_path = sorted(Path(dir + '/policies').glob(file_type))[0]
        os.remove(policies_history_path)
        enemy_policies_history_path = sorted(Path(dir + '/auxiliary_policies').glob(file_type))[0]
        os.remove(enemy_policies_history_path)
        value_history_path = sorted(Path(dir + '/value').glob(file_type))[0]
        os.remove(value_history_path)
        time.sleep(0.5)

def del_all_history(dir, file_type):
    print('Delete History data.')
    count = len(sorted(Path(dir + '/board').glob(file_type)))
    for _ in tqdm(range(count)):
        board_history_path = sorted(Path(dir + '/board').glob(file_type))[0]
        os.remove(board_history_path)
        policies_history_path = sorted(Path(dir + '/policies').glob(file_type))[0]
        os.remove(policies_history_path)
        enemy_policies_history_path = sorted(Path(dir + '/auxiliary_policies').glob(file_type))[0]
        os.remove(enemy_policies_history_path)
        value_history_path = sorted(Path(dir + '/value').glob(file_type))[0]
        os.remove(value_history_path)
        time.sleep(0.5)

def copy_dir(iter, path = './c_model'):
    if(int(iter) > 0):
        source_folder = './c_model'
        destination_folder = './save_model/model' + str(iter)
        shutil.copytree(source_folder, destination_folder)
        print("copy to " + destination_folder)
    else:
    # fetch all files        
        destination_folder = path
        source_folder = './c_model/'
        for file_name in os.listdir(source_folder):
            # construct full file path
            source = source_folder + file_name
            destination = destination_folder + file_name
            if os.path.isdir(source):
                #copy dir
                for sub_file_name in os.listdir(source):
                    sub_source = source + '/' + sub_file_name
                    sub_destination = destination + '/' + sub_file_name
                    if(os.path.isfile(sub_source)):
                        shutil.copy(sub_source, sub_destination)
                        #print('copied', sub_file_name)
            else:
                # copy only files
                if os.path.isfile(source):
                    shutil.copy(source, destination)
                    #print('copied', file_name)
        print(path, "copy model complete!")

def getSymmetries(my_board, opp_board, pi, nxt_pi, val):
    pi_board = np.reshape(pi[:], (15, 15))
    nxt_pi_board = np.reshape(nxt_pi[:], (15, 15))
    l = []
    for i in range(1, 5):
        for j in [True, False]:
            newmy = np.rot90(my_board, i)
            newopp = np.rot90(opp_board, i)
            newPi = np.rot90(pi_board, i)
            newNxtPi = np.rot90(nxt_pi_board, i)
            if j:
                newmy = np.fliplr(newmy)
                newopp = np.fliplr(newopp)
                newPi = np.fliplr(newPi)
                newNxtPi = np.fliplr(newNxtPi)
            l += [(newmy, newopp, list(newPi.ravel()), list(newNxtPi.ravel()), val)]
    return l

def expand_data(boards, polices, auxiliary_policies, values):
    expanded_boards = []
    expanded_polices = []
    expanded_auxiliary_policies = []
    expanded_values = []
    for board, policy, auxiliary_policy, value in zip(boards, polices, auxiliary_policies, values):
        expanded_data = getSymmetries(board[0], board[1], policy, auxiliary_policy, value)
        for newmy, newopp, newPi, newNxtPi, val in expanded_data:
            expanded_boards += [[newmy, newopp]]
            expanded_polices += [newPi]
            expanded_auxiliary_policies += [newNxtPi]
            expanded_values += [val]
    expanded_boards = np.array(expanded_boards).transpose(0,2,3,1).astype('float32')
    expanded_polices = np.array(expanded_polices).astype('float32')
    expanded_auxiliary_policies = np.array(expanded_auxiliary_policies).astype('float32')
    expanded_values = np.array(expanded_values).astype('float32')
    # print(np.shape(expanded_values))
    # print(np.shape(expanded_boards))
    return expanded_boards, expanded_polices, expanded_auxiliary_policies, expanded_values