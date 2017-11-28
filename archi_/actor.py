import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell, DropoutWrapper
distr = tf.contrib.distributions
from tqdm import tqdm
import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams


# Tensor summaries for TensorBoard visualization
def variable_summaries(name,var, with_max_min=False):
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    if with_max_min == True:
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))


############################
### DEFINING THE NETWORK ###
############################


class Actor(object):

    def __init__(self, config):
        self.batch_size = config.batch_size # batch size
        self.a_steps = config.a_steps # length of melody (listen)
        self.b_steps = config.b_steps # length of melody/harmony (attend & play)
        self.num_roots = config.num_roots # dimension of a keyboard
        self.threshold = config.threshold # for playing a note
        self.hidden_dimension = config.hidden_dimension # for melody / keyboard embedding, RNN modules...

        self.is_training = config.is_training # mode
        self.initializer = tf.contrib.layers.xavier_initializer() # variables initializer
        self.temperature = config.temperature

        self.global_step = tf.Variable(0, trainable=False, name="global_step") # global step
        self.lr1_start = config.lr1_start # initial learning rate
        self.lr1_decay_rate = config.lr1_decay_rate # learning rate decay rate
        self.lr1_decay_step = config.lr1_decay_step # learning rate decay step

        # input and target of the model
        self.input = tf.placeholder(tf.float32, [None, self.a_steps+self.b_steps, self.num_roots+1+1]) # melody input [Batch Size, k * Sequence Length, Keyboard_dim]       +1 +1 for silence & on_kick ft
        self.target = tf.placeholder(tf.float32, [None, self.b_steps, self.num_roots+1]) # target chords [Batch Size, Sequence Length, Keyboard_dim]        +1 for silence
        
        with tf.variable_scope("encoder"):
            # Encode melody
            self.encode_melody()
            # Encode instrument (piano)
            self.encode_instru()
        with tf.variable_scope("decoder"):
            # Decode accompaniement
            self.listen()
            self.attend_n_play()
        with tf.variable_scope("training"):
            # Training
            self.train()
        self.merged = tf.summary.merge_all()

    def encode_melody(self):
        # Embed input sequence
        W_embed =tf.get_variable("weights", [1,self.num_roots+1+1,self.hidden_dimension])
        embedded_input = tf.nn.conv1d(self.input, W_embed, 1, "VALID", name="embedded_input")
        # Batch Normalization
        embedded_input = tf.layers.batch_normalization(embedded_input, axis=2, training=self.is_training, name='layer_norm', reuse=None)
        self.melody = tf.transpose(embedded_input,[1,0,2]) # [Time steps, batch size, h_dim]
 
    def encode_instru(self):
        # Keyboard Representation
        positional_keyboard = tf.tile(tf.expand_dims(tf.range(self.num_roots+1), 0),[self.batch_size, 1])
        lookup_table = tf.get_variable('lookup_table', dtype=tf.float32, shape=[self.num_roots+1, self.hidden_dimension])
        self.keyboard_embedding = tf.nn.embedding_lookup(lookup_table, positional_keyboard, name="positional_encoding")

    def listen(self):
        # Decoder LSTM cell        
        self.cell1 = LSTMCell(self.hidden_dimension, initializer=self.initializer) # Melody LSTM cell (listen + attend_n_play)
        self.cell2 = LSTMCell(self.hidden_dimension, initializer=self.initializer) # Harmony LSTM cell (attend_n_play)

        # Pointing mechanism
        with tf.variable_scope("pointer"):
            self.W_ref =tf.get_variable("W_ref",[1,self.hidden_dimension,self.hidden_dimension],initializer=self.initializer)
            self.W_q =tf.get_variable("W_q",[self.hidden_dimension,self.hidden_dimension],initializer=self.initializer)
            self.v =tf.get_variable("v",[self.hidden_dimension],initializer=self.initializer)

            self.W1 =tf.get_variable("W1",[self.num_roots+1,self.hidden_dimension],initializer=self.initializer)
            self.b1 =tf.get_variable("b1",[self.hidden_dimension],initializer=None)
            self.W2 =tf.get_variable("W2",[self.hidden_dimension,self.num_roots+1],initializer=self.initializer)
            self.b2 =tf.get_variable("b2",[self.num_roots+1],initializer=None)

        # Loop the decoding process and collect results
        self.s = tf.zeros([self.batch_size,self.hidden_dimension]), tf.zeros([self.batch_size,self.hidden_dimension]) # Melody initial (LSTM) state
        for step in range(self.a_steps):
            if step > 0:
                tf.get_variable_scope().reuse_variables()
            # Run the cell on a combination of the input and state
            _, self.s = self.cell1(self.melody[step],self.s)

    def attend_n_play(self):
        # Output = Distribution over keyboard for music generation
        self.pointing_ = []

        # From a query (decoder output) [Batch size, n_hidden], predict a distribution over a set of reference vectors (Keyboard representation) [Batch size, seq_length, n_hidden]
        def attention(query):
            encoded_ref = tf.nn.conv1d(self.keyboard_embedding, self.W_ref, 1, "VALID", name="encoded_ref") # [Batch size, seq_length, n_hidden]
            encoded_query = tf.expand_dims(tf.matmul(query, self.W_q, name="encoded_query"), 1) # [Batch size, 1, n_hidden]
            scores = tf.reduce_sum(self.v * tf.tanh(encoded_ref + encoded_query), [-1], name="scores") # [Batch size, seq_length]
            scores = 10.0*tf.tanh(scores) # control entropy
            pointing_0 = tf.nn.softmax(scores, name="attention") # Pointer: [Batch size, Seq_length]
            return pointing_0

        # Loop the decoding process and collect results
        self.s_ = tf.zeros([self.batch_size,self.hidden_dimension]), tf.zeros([self.batch_size,self.hidden_dimension]) # Harmony initial (LSTM) state
        for step in range(self.b_steps):
            if step > 0:
                tf.get_variable_scope().reuse_variables()
            # Run the cell on a combination of the input and state
            output, self.s = self.cell1(self.melody[self.a_steps+step],self.s)
            # Attention mechanism
            pointing_0 = attention(output)
            # Additional RNN layer on pointing mechanism (piano) - because chords are non linear + recurrent relation
            pointing_1, self.s_ = self.cell2(tf.matmul(pointing_0, self.W1)+self.b1,self.s_)
            # Dense layer for final chord prediction
            pointing_2 = tf.tanh(tf.matmul(pointing_1, self.W2)+self.b2)
            if self.is_training == False:
                pointing_2 = pointing_2/self.temperature # control diversity of sampling (inference mode)
            pointing = tf.nn.softmax(10.0*pointing_2, name="attention") # Pointer: [Batch size, Seq_length]
            self.pointing_.append(pointing)

        # Stack pointing distribution
        self.pointing_ = tf.stack(self.pointing_ ,axis=1) # [Batch,seq_length,Keyboard_dim]

        # Hard sample
        self.target_chords = tf.cast(self.target,tf.int32)
        self.played_chords = tf.cast(tf.greater_equal(self.pointing_,self.threshold),tf.int32)
        # Accuracy = ...
        errors = tf.cast(tf.minimum(tf.reduce_sum(tf.abs(self.target_chords-self.played_chords),2),1),tf.float32) # [batch_size,seq_length] 	0 if good chord, 1 if error.
        accuracies = 1-tf.reduce_mean(errors,1)
        variable_summaries('accuracy',accuracies, with_max_min = True)
        self.accuracy = tf.reduce_mean(accuracies,0)


    def train(self):

        # Update moving_mean and moving_variance for batch normalization layers
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            # Actor learning rate
            self.lr1 = tf.train.exponential_decay(self.lr1_start, self.global_step, self.lr1_decay_step,self.lr1_decay_rate, staircase=False, name="learning_rate1")
            tf.summary.scalar('lr',self.lr1)
            # Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr1,beta1=0.9,beta2=0.99, epsilon=0.0000001)
                
            # Rescale pointing_distribution (sum = 1) by number of target notes at each step, for each batch
            scale = tf.cast(tf.count_nonzero(self.target,axis=2,keep_dims=True),tf.float32)
            # Loss function = CROSS ENTROPY (supervised)
            self.cross_entropy = -tf.reduce_sum(self.target * tf.log(tf.clip_by_value(scale*self.pointing_,1e-10,4.0)),2) # cross entropy at each output step (Sum on Keyboard_dim)
            self.cross_entropy = tf.reduce_mean(self.cross_entropy,1) # mean entropy for each seq prediction (Mean on steps)
            variable_summaries('cross_entropy',self.cross_entropy, with_max_min = True)
            self.cross_entropy = tf.reduce_mean(self.cross_entropy,0) # mean entropy % batch

            # Minimize step
            gvs = optimizer.compute_gradients(self.cross_entropy)
            capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs if grad is not None] # L2 clip
            self.minimize = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)