from dn5 import DN
#from dn10 import DN
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError, KLDivergence
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Softmax
from tensorflow.keras import backend as K

import keras.backend as K
from keras.callbacks import Callback
from keras.layers import BatchNormalization
from callback import create_swa_callback_class

import numpy as np
from datetime import datetime
from pathlib import Path
import pickle
import os
import time
import sys
from utils import *
args = dotdict({
    'process_num' : 8,
    'simulation_count' : 4096,
    #path
    'original_data_path' : './original_train_data',
    'original_file_type' : '*.history',
    'rotated_data_path' : './rotated_train_data',
    'rotated_file_type' : '*.npy',
    #training data sampling
    'sampling_freq' : 16, 
    'replay_buffer_size' : 40960 * 16, #40960 * 16
    #sliding window
    'start_window_idx' : 4,
    #SVG
    'lr' : 0.1,
    'momentum' : 0.9,
    'decay' : 0.00003,
    #method
    'method' : 'gumbel',
    'batch_size' : 2048,
    'sampling_epoch_num' : 16,
    'use_sample' : True,
    #swa
    'use_swa' : False,
    'start_epoch' : 2, #must >= 2 從第二個開始
    'swa_batch_size' : 2048 ,
    'swa_freq' : 2, # must >=2 每兩個做一次swa
    'swa_epoch_num' : 10, 
})

def MeanSquaredError_15(y_true, y_pred):
    return 1.5*MeanSquaredError()(y_true, y_pred)

def Soft_max_KLDivergence_015(y_true, y_pred):
    return 0.15*KLDivergence()(Softmax()(y_true), Softmax()(y_pred))

def Soft_max_KLDivergence(y_true, y_pred):
    return KLDivergence()(Softmax()(y_true), Softmax()(y_pred))

def MeanSquaredError_15(y_true, y_pred):
    return 1.5*MeanSquaredError()(y_true, y_pred)

def Soft_max_Crossentropy_015(y_true, y_pred):
    return 0.15*CategoricalCrossentropy()(Softmax()(y_true), Softmax()(y_pred))

def Soft_max_Crossentropy(y_true, y_pred):
    return CategoricalCrossentropy()(Softmax()(y_true), Softmax()(y_pred))

def load_training_data(args):
    print('Start loading training data....')
    #load latest self-play data
    if(len(os.listdir(args.original_data_path + '/board')) == len(os.listdir(args.original_data_path + '/policies')) and len(os.listdir(args.original_data_path + '/policies')) == len(os.listdir(args.original_data_path + '/auxiliary_policies')) and len(os.listdir(args.original_data_path + '/auxiliary_policies')) == len(os.listdir(args.original_data_path + '/value'))):
        original_data_size = len(os.listdir(args.original_data_path + '/board'))
    else:
        print("Original train data count has some problem.")
        exit()
    print('Original train data file count is:', original_data_size , '/', args.process_num, '=', original_data_size / int(args.process_num))
    if (original_data_size > 0):
        original_boards = load_data(args.original_data_path + '/board', original_data_size, file_type = args.original_file_type)
        original_policies = load_data(args.original_data_path + '/policies', original_data_size, file_type = args.original_file_type)
        original_auxiliary_policies = load_data(args.original_data_path + '/auxiliary_policies', original_data_size, file_type = args.original_file_type)
        original_values = load_data(args.original_data_path + '/value', original_data_size, file_type = args.original_file_type)

        #reshape data
        original_boards = np.array(original_boards).reshape(len(original_boards), 2, 15, 15)
        original_policies = np.array(original_policies).reshape(len(original_policies), 15, 15)
        original_auxiliary_policies = np.array(original_auxiliary_policies).reshape(len(original_auxiliary_policies), 15, 15)
        original_values = np.array(original_values)
        print('Original train data shape: ')
        print(original_boards.shape, original_policies.shape, original_auxiliary_policies.shape, original_values.shape)
        
        #expand data
        print('Start expand original training data.')
        st = time.time()
        expand_boards = expand_board_rotated_data(original_boards)
        expand_policies = expand_policy_rotated_data(original_policies)
        expand_auxiliary_policies = expand_policy_rotated_data(original_auxiliary_policies)
        expand_values = expand_value_rotated_data(original_values)
        ed = time.time()
        print('Expand data completed.\nCost time:', ed - st)

    #load ex-expanded data
    if(len(os.listdir(args.rotated_data_path + '/board')) == len(os.listdir(args.rotated_data_path + '/policies')) and len(os.listdir(args.rotated_data_path + '/policies')) == len(os.listdir(args.rotated_data_path + '/auxiliary_policies')) and len(os.listdir(args.rotated_data_path + '/auxiliary_policies')) == len(os.listdir(args.rotated_data_path + '/value'))):
        rotated_data_size = len(os.listdir(args.rotated_data_path + '/board'))
    else:
        print("Rotated train data count has some problem.")
        exit()
    print('Rotated train data file count is:', rotated_data_size)
    if (rotated_data_size > 0):
        rotated_boards = load_data(args.rotated_data_path + '/board', rotated_data_size, file_type = args.rotated_file_type)
        rotated_policies = load_data(args.rotated_data_path + '/policies', rotated_data_size, file_type = args.rotated_file_type)
        rotated_auxiliary_policies = load_data(args.rotated_data_path + '/auxiliary_policies', rotated_data_size, file_type = args.rotated_file_type)
        rotated_values = load_data(args.rotated_data_path + '/value', rotated_data_size, file_type = args.rotated_file_type)

    #combine latest self-play data and ex-data
    if (original_data_size == 0 or rotated_data_size == 0):
        if(original_data_size > 0 and rotated_data_size == 0):
            y_boards = expand_boards
            y_policies = expand_policies
            y_next_policies = expand_auxiliary_policies
            y_values = expand_values
        elif(rotated_data_size > 0 or original_data_size == 0):
            y_boards = rotated_boards
            y_policies = rotated_policies
            y_next_policies = rotated_auxiliary_policies
            y_values = rotated_values
        else:
            print("Have not any training data file.")
            exit()
    else:
        y_boards = np.concatenate([rotated_boards, expand_boards])
        y_policies = np.concatenate([rotated_policies, expand_policies])
        y_next_policies = np.concatenate([rotated_auxiliary_policies, expand_auxiliary_policies])
        y_values = np.concatenate([rotated_values, expand_values])

    #save 
    if (original_data_size > 0):
        print('Save expanded training data.')
        now_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        np.save(args.rotated_data_path + '/board/' + now_time, expand_boards)
        np.save(args.rotated_data_path + '/policies/' + now_time, expand_policies)
        np.save(args.rotated_data_path + '/auxiliary_policies/' + now_time, expand_auxiliary_policies)
        np.save(args.rotated_data_path + '/value/' + now_time, expand_values)

        del_history(args.original_data_path, args.process_num)

    print('Total training data: ', len(y_boards))
    print(y_boards.shape, y_policies.shape, y_next_policies.shape, y_values.shape)
    print('Loading training data completed....')
    return y_boards, y_policies, y_next_policies, y_values

def train_network(args, iter):
    xs, y_policies, y_next_policies, y_values = load_training_data(args)
    model = DN()
    if(os.path.exists('./c_model/')):
        model.load_weights('./c_model/')
        print('loading best model')
    else:
        print('Can not load model')
        exit()
    
    opt = SGD( 
        learning_rate=args.lr,
        momentum=args.momentum,
        decay=args.decay,
    )

    if (args.method == 'katago'):
        model.compile(loss=[Soft_max_Crossentropy, Soft_max_Crossentropy_015, MeanSquaredError_15],optimizer = opt)
    elif (args.method == 'gumbel'):
        model.compile(loss=[Soft_max_KLDivergence, Soft_max_KLDivergence_015, MeanSquaredError_15],optimizer = opt)
    else:
        print('Method error.')
        exit()
    
    if args.use_swa == True:
        print('use swa')
        SWA = create_swa_callback_class(K, Callback, BatchNormalization)
        callback = SWA(start_epoch=args.start_epoch, 
                lr_schedule='cyclic', 
                swa_lr=args.lr * 0.01,
                swa_lr2=args.lr,
                swa_freq=args.swa_freq,
                batch_size=args.swa_batch_size, # needed when using batch norm
                verbose=1)
        epoch_num = args.swa_epoch_num
    elif args.use_sample == True:
        print('use sample')
        epoch_num = 1
        callback = LambdaCallback(
                on_epoch_begin=lambda epoch,logs:
                        print('\rTrain {}/{}'.format(epoch + 1, epoch_num), end=''))
    else:
        print('use normal')
        epoch_num = args.swa_epoch_num
        callback = LambdaCallback(
                on_epoch_begin=lambda epoch,logs:
                        print('\rTrain {}/{}'.format(epoch + 1, epoch_num), end=''))
        
    
    model.build(input_shape = (None,15,15,2))
    replay_buffer_current_size = xs.shape[0]
    print('Current buffer size:', replay_buffer_current_size)
    if args.use_swa == True:
        his = model.fit(xs, [y_policies, y_next_policies, y_values], shuffle = True, batch_size=args.batch_size, epochs=epoch_num , verbose=1, callbacks=[callback], workers=4)
    elif args.use_sample == True:
        sampling_num = int(y_values.shape[0] / args.sampling_freq)
        for i in range(args.sampling_epoch_num):
            print('Sampling',i + 1,':')
            idx = np.random.choice(replay_buffer_current_size, size=sampling_num, replace = False)
            train_xs, train_y_policies, train_y_next_policies, train_y_values = xs[idx], y_policies[idx], y_next_policies[idx], y_values[idx]
            his = model.fit(train_xs, [train_y_policies, train_y_next_policies, train_y_values], shuffle = True, batch_size=args.batch_size, epochs=epoch_num , verbose=1, callbacks=[callback], workers=4)
    else:
        his = model.fit(xs, [y_policies, y_next_policies, y_values], shuffle = True, batch_size=args.batch_size, epochs=epoch_num , verbose=1, callbacks=[callback], workers=4)
    
    model.save_weights('./c_model/')
    model.save('./c_model/', save_format = 'tf')

    if (iter % 5 == 0):
        print("./save_model/", iter)
        copy_dir(iter)

    end_window_idx = (args.replay_buffer_size - args.start_window_idx * 8 * args.simulation_count) / (8 * args.simulation_count) * 2 + args.start_window_idx - 1
    if (iter >= args.start_window_idx and iter <= end_window_idx and iter % 2 == 0):
        del_history(args.rotated_data_path, count = 1, file_type = args.rotated_file_type)

    if (y_values.shape[0] >= args.replay_buffer_size):
        del_history(args.rotated_data_path, count = 1, file_type = args.rotated_file_type)
    
    K.clear_session()
    del model

def save_init_mode():
    model = DN()#.model()
    model.build(input_shape = (None,15,15,2))
    print(model.summary())
    
    x = np.array([0]*(2*15*15)).astype('float32')
    x = x.reshape(1,2,15,15).transpose(0,2,3,1).astype('float32')

    #利用模型的預測去取得'策略'與'局勢價值'
    y = model.predict(x, batch_size=1)
    model.save('./c_model/', save_format = 'tf')
    model.save_weights('./c_model/', save_format = 'tf')
    copy_dir(-1, './init_model/')
    copy_dir(-1, './save_model/model1/')
    copy_dir(-1, './p_model/')
    K.clear_session()
    del model
    
def test_train():
    train_data_size = 0
    self_play_count = 0
    end_window_idx = (args.replay_buffer_size - args.start_window_idx * 8 * args.simulation_count) / (8 * args.simulation_count) * 2 + args.start_window_idx - 1
    for i in range(1, 100):
        train_data_size += args.simulation_count * 8
        self_play_count += 1
        print(i, train_data_size, self_play_count)
        if (i >= args.start_window_idx and i <= end_window_idx and i % 2 == 0):
            train_data_size -= args.simulation_count * 8
            self_play_count -= 1

        if (train_data_size >= args.replay_buffer_size):
            train_data_size -= args.simulation_count * 8
            self_play_count -= 1
    print(args.replay_buffer_size)

if __name__ == '__main__':
    iter = int(sys.argv[1])
    st = time.time()
    #test_train()
    print(iter, 'training start')
    train_network(args, iter)
    # save_init_mode()
    #del_all_history(args.original_data_path, args.original_file_type)
    #del_all_history(args.rotated_data_path, args.rotated_file_type)
    ed = time.time()
    print('Training cost time:',ed - st)


        