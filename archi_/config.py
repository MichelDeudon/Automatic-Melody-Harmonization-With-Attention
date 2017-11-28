#-*- coding: utf-8 -*-
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')



# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch_size', type=int, default=64, help='batch size')																	##################### switch to 1 for test
data_arg.add_argument('--a_steps', type=int, default=12, help='length of melody (listen)')
data_arg.add_argument('--b_steps', type=int, default=12, help='length of melody/harmony (attend & play)')
data_arg.add_argument('--num_roots', type=int, default=24, help='dimension of a keyboard')

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--hidden_dimension', type=int, default=128, help='for melody / keyboard embedding, RNN modules...')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--nb_epoch', type=int, default=40, help='nb epoch')
train_arg.add_argument('--lr1_start', type=float, default=0.001, help='actor learning rate')
train_arg.add_argument('--lr1_decay_step', type=int, default=5000, help='lr1 decay step')
train_arg.add_argument('--lr1_decay_rate', type=float, default=0.96, help='lr1 decay rate')
train_arg.add_argument('--threshold', type=float, default=0.15, help='for playing a note')
train_arg.add_argument('--temperature', type=float, default=1., help='pointer_net initial temperature')

# Misc
misc_arg = add_argument_group('User options') #####################################################
misc_arg.add_argument('--is_training', type=str2bool, default=True, help='switch to inference mode once trained')								##################### switch to False for test
misc_arg.add_argument('--restore_model', type=str2bool, default=False, help='whether or not model is retrieved')								##################### switch to True for test

misc_arg.add_argument('--mode', type=str, default='minor', help='major or minor')
misc_arg.add_argument('--train_folder', type=str, default='train', help='folder containing training tracks') 
misc_arg.add_argument('--test_folder', type=str, default='test', help='folder containing testing tracks') 

misc_arg.add_argument('--log_dir', type=str, default='summary/minor', help='summary writer log directory') 



def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed