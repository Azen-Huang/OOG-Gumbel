#from DN_5 import DN
#from DN_10 import DN
from dn5 import DN
import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
#from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.losses import KLDivergence, MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Softmax
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import pickle
import os
import time
import sys
import multiprocessing
from swa.keras import SWA

RN_EPOCHS = 5 #訓練次數
BATCH_SIZE = 4096
'''
step 1:load data(history)

step 2:重塑訓練資料的shape
dual network的輸入shape為（3,3,2）
但這邊我們要一次專遞躲避訓練資料，因此要知其成為4軸陣列（訓練資料筆數,3,3,2） 本例要對500個訓練資料進行預測，所以重塑為(500,3,3,2)

step 3:載入最佳玩家的模型
將最佳玩家的模型作為訓練最新玩家模型的初始狀態
雖然dual_network.py還處於未訓練狀態
但一開始還是先將其載入作為最佳玩家，之後再將訓練後的模型輸出為最新玩家

step 4:編譯模型 編譯模型時要制定loss func，optimizer，評估指標
loss func：
    策略為分類問題所以用 categorical_crossentropy
    局勢價值為回歸問題所以用mse
Optimizer：Adam
評估指標：因為訓練dual network不會進行驗證，所以這裡先不用

step 5:調整學習率 利用LearningRateScheduler
預設為0.001，經過50步降為0.005 80步降為0.00025

step 6:印出訓練次數
利用callback輸出訓練次數

step 7:進行訓練

step 8:儲存最新玩家的模型
'''
def rotate(A):
    B = np.copy(A)
    B[:] = [[row[i] for row in B[::-1]] for i in range(len(B))]
    return B

def mirror(A):
    B = np.copy(A)
    return np.flip(B, axis = 1)

def expand_xs_rotated_data(xs):
    print('Start expand board.')
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
    print('Expand board finish.')
    return [0, xs]

def expand_policy_rotated_data(policies, i):  
    #print(policies[9],'\n')
    if (i == 1):
        print('Start expand policy.')
    else:
        print('Start expand auxiliary policy')
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
    if (i == 1):
        print('Expand policy finish.')
    else:
        print('Expand auxiliary policy finish.')
    return [i, policies]

def expand_value_rotated_data(value):
    print('Start expand value.')
    copy_value = np.copy(value)
    value = np.concatenate([value, copy_value])
    value = np.concatenate([value, copy_value])
    value = np.concatenate([value, copy_value])
    value = np.concatenate([value, copy_value])
    value = np.concatenate([value, copy_value])
    value = np.concatenate([value, copy_value])
    value = np.concatenate([value, copy_value])
    value = value.astype('float32')
    print('Expand value finish.')
    return [3, value]
    
def MeanSquaredError_15(y_true, y_pred):
    return 1.5*MeanSquaredError()(y_true, y_pred)

def KLDivergence_015(y_true, y_pred):
    return 0.15*KLDivergence()(Softmax()(y_true), Softmax()(y_pred))

def Soft_max_KLDivergence(y_true, y_pred):
    return KLDivergence()(Softmax()(y_true), Softmax()(y_pred))

def drawlearning(history):
    import matplotlib.pyplot as plt
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['loss'])
    #plt.plot(history.history['output_1_loss'])
    #plt.plot(history.history['output_2_loss'])
    #plt.plot(history.history['output_3_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'policy loss','value loss'], loc='upper left') 
    plt.show()
    plt.savefig("lr_2.png")


dir = './train_data'
def load_data(path,i): #載入self_play所儲存的history檔
    history_path = sorted(Path(path).glob('*.history'))[i]
    with history_path.open(mode = 'r') as f:
        vec = f.readlines()
        f.close()
    
    output_arr = []
    for v in vec:
        v = v.split(", ")
        output_arr.append(v[:-1])
    return output_arr


def del_hitory():
    board_history_path = sorted(Path(dir + '/board').glob('*.history'))[0]
    policies_history_path = sorted(Path(dir + '/policies').glob('*.history'))[0]
    auxiliary_policies_history_path = sorted(Path(dir + '/auxiliary_policies').glob('*.history'))[0]
    value_history_path = sorted(Path(dir + '/value').glob('*.history'))[0]
    
    os.remove(board_history_path)
    os.remove(policies_history_path)
    os.remove(auxiliary_policies_history_path)
    os.remove(value_history_path)

def train_network(thread_num):
    #step 1 載入訓練資料
    if(len(os.listdir(dir + '/board')) == len(os.listdir(dir + '/policies')) and len(os.listdir(dir + '/policies')) == len(os.listdir(dir + '/auxiliary_policies')) and len(os.listdir(dir + '/auxiliary_policies')) == len(os.listdir(dir + '/value'))):
        SIZE = len(os.listdir(dir + '/board'))
    else:
        print("file count has some problem.")
        exit()

    print('File count is:', SIZE , '/', thread_num, '=', SIZE / int(thread_num))
    boards = load_data(dir + '/board',0)
    policies = load_data(dir + '/policies',0)
    auxiliary_policies = load_data(dir + '/auxiliary_policies',0)
    value = load_data(dir + '/value',0)[0]
    
    for i in range(1, SIZE):
        boards += load_data(dir + '/board',i)
        policies += load_data(dir + '/policies',i)
        auxiliary_policies += load_data(dir + '/auxiliary_policies',i)
        value += load_data(dir + '/value',i)[0]
    
    
    print(len(boards), len(policies), len(auxiliary_policies), len(value))
    # #step 2:重塑訓練資料的shape
    xs = np.array(boards).reshape(len(boards), 2, 15, 15)
    y_policies = np.array(policies).reshape(len(policies), 15, 15)
    y_next_policies = np.array(auxiliary_policies).reshape(len(auxiliary_policies), 15, 15)
    y_values = np.array(value)

    print(xs.shape, y_policies.shape, y_next_policies.shape, y_values.shape)
    start_time = time.time()
    print("Start expand data.")

    queue = multiprocessing.Queue()
    processes = []

   
    p1 = multiprocessing.Process(target=lambda q, arg1: q.put(expand_xs_rotated_data(arg1)), args=(queue, xs))
    processes.append(p1)   

    p2 = multiprocessing.Process(target=lambda q, arg1, arg2: q.put(expand_policy_rotated_data(arg1, arg2)), args=(queue, y_policies, 1))
    processes.append(p2)
    
    p3 = multiprocessing.Process(target=lambda q, arg1,  arg2: q.put(expand_policy_rotated_data(arg1, arg2)), args=(queue, y_next_policies, 2))
    processes.append(p3)
    
    p4 = multiprocessing.Process(target=lambda q, arg1: q.put(expand_value_rotated_data(arg1)), args=(queue, y_values))
    processes.append(p4)
    
    for process in processes:
        process.start()

    finish_count = 0
    while (finish_count < 4):
        while not queue.empty():
            finish_count += 1
            result = queue.get()
            if(result[0] == 0):
                xs = result[1]
                processes[0].join()
            elif(result[0] == 1):
                y_policies = result[1]
                processes[1].join()
            elif(result[0] == 2):
                y_next_policies = result[1]
                processes[2].join()
            elif(result[0] == 3):
                y_values = result[1]
                processes[3].join()

    print('All process done.')
    print(xs.shape, y_policies.shape, y_next_policies.shape, y_values.shape)
    print("expand data complete.")
    end_time = time.time()
    print("Expand data time taken: ", end_time - start_time)
    #step 3 載入最佳玩家的模型
    #model = DN().model()
    model = DN()
    if(os.path.exists('./c_model/')):
        model.load_weights('./c_model/')
        #model.load_weights('./c_model/latest.h5')
        print('loading best model')
    else:
        print('Can not load model')
        exit()
    
    opt = SGD( 
        learning_rate=0.1,
        momentum=0.9,
        decay=0.00003
    )
    opt = tfa.optimizers.SWA(opt, start_averaging = 1, average_period = 1)
    #opt = Adam(learning_rate=1e-3)
    model.compile(loss=[Soft_max_KLDivergence, KLDivergence_015, MeanSquaredError_15], optimizer=opt)
    #model.compile(loss=[Soft_max_KLDivergence, KLDivergence_015, MeanSquaredError_15],optimizer='adam')
    #model.compile(loss=['categorical_crossentropy',CategoricalCrossentropy_015,'mse'],optimizer=opt)
    model.build(input_shape = (None,15,15,2))

    #print(model.summary())
    '''
    def step_decay(epoch):
            x = 0.0001
            if epoch >= 10: x = 0.00007
            if epoch >= 15: x = 0.00005
            return x
    '''
    '''
    def step_decay(epoch):
            x = 0.0001
            if epoch >= 4: x = 0.00008
            if epoch >= 8: x = 0.00006
            if epoch >= 12: x = 0.00005
            if epoch >= 16: x = 0.00004
            return x
    '''

    # def step_decay(epoch):
    #         x = 0.0001
    #         if epoch >= 4: x = 0.00008
    #         if epoch >= 8: x = 0.00006
    #         return x
        
    # lr_decay = LearningRateScheduler(step_decay)

    # define swa callback
    start_epoch = 1
    swa = SWA(start_epoch=start_epoch, 
            lr_schedule='cyclic', 
            swa_lr=0.05,
            swa_lr2=0.01,
            swa_freq=1,
            batch_size=BATCH_SIZE, # needed when using batch norm
            verbose=1)
    
    print_callback = LambdaCallback(
                on_epoch_begin=lambda epoch,logs:
                        print('\rTrain {}/{}'.format(epoch + 1,RN_EPOCHS), end=''))
    
    his = model.fit(xs, [y_policies, y_next_policies, y_values], shuffle = True, batch_size=BATCH_SIZE, epochs=RN_EPOCHS, verbose=1, callbacks=[swa], workers=4)
    #his = model.fit(xs, [y_policies, y_next_policies, y_values], shuffle = True, batch_size=BATCH_SIZE, epochs=RN_EPOCHS, verbose=1, callbacks=[lr_decay, print_callback], workers=4)
    #his = model.fit(xs, [y_policies, y_next_policies, y_values], shuffle = True, batch_size=512, epochs=RN_EPOCHS, verbose=1, callbacks=[print_callback], workers=4)
    #his = model.fit(xs, [y_policies, y_next_policies, y_values], batch_size=512, epochs=RN_EPOCHS, callbacks=[print_callback], verbose=1, workers=4)
    model.save_weights('./c_model/')
    model.save('./c_model/', save_format = 'tf')
    K.clear_session()
    del model
    #drawlearning(his)

def save_init_mode():
    KataGo_model = DN()#.model()

    #KataGo_model.compile(loss=[Soft_max_KLDivergence,KLDivergence_015,MeanSquaredError_15],optimizer='adam')
    KataGo_model.build(input_shape = (None,15,15,2))
    print(KataGo_model.summary())
    
    x = np.array([0]*(2*15*15)).astype('float32')
    x = x.reshape(1,2,15,15).transpose(0,2,3,1).astype('float32')
    # #利用模型的預測去取得'策略'與'局勢價值'
    y = KataGo_model.predict(x,batch_size=1)

    KataGo_model.save('./c_model/', save_format = 'tf')
    KataGo_model.save_weights('./c_model/', save_format = 'tf')
    #KataGo_model.save_weights('./c_model/latest.h5')
    K.clear_session()
    del KataGo_model

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    

    start_time = time.time()
    thread_num = 1
    try:
        print("Self play have ", sys.argv[1], "Process")
        thread_num = sys.argv[1]
    except IndexError as e:
        # Memory growth must be set before GPUs have been initialized
        print("No muti-Process!!")
    #save_init_mode()
    train_network(thread_num)
    '''
    del_hitory()
    for i in range(19):
        del_hitory()
    '''
    end_time = time.time()

    print("Time taken: ", end_time - start_time)
    