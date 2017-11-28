import tensorflow as tf
from tqdm import tqdm
import random
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

from actor import Actor
from config import get_config
import data_loader



##############################
### Keyboard Vizualization ###
##############################

# Heatmap of attention
def visualize_attention(matrix):
    # plot heatmap
    fig = plt.figure()
    plt.imshow(np.transpose(matrix), interpolation='nearest', cmap='hot')
    plt.colorbar()
    plt.title('Pointing distribution')
    plt.ylabel('Notes')
    plt.xlabel('Step t')
    plt.show()


#################
### Run GRAPH ###
#################

# Get running configuration
config, _ = get_config()

# Build tensorflow graph from config
print("Building graph...")
Bob = Actor(config)

# Saver to save & restore all the variables.
variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)

print("Starting session...")
with tf.Session() as sess:
    # Run initialize op
    sess.run(tf.global_variables_initializer())

    # Restore variables from disk.
    if config.restore_model==True:
        saver.restore(sess, "save/"+config.mode+"/actor.ckpt")
        print("Model restored.")

    # Create data generator
    dataset = data_loader.DataGenerator(a_steps=config.a_steps, b_steps=config.b_steps)

    # Load the dataset
    dataset.process_files(folder=config.train_folder,mode=config.mode)
    # X = np.load(config.mode+"_"+config.folder+"_X.npy").item()     # load X,y as .npy if running on Helios or whatever 
    # y = np.save(config.mode+"_"+config.folder+"_y.npy").item()     # (don't need to repreprocess all the data)
    X, y = dataset.training_seq, dataset.target_seq

    # Split into training / testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_test, y_test = np.asarray(X_test), np.asarray(y_test)

    # Summary writer
    writer = tf.summary.FileWriter(config.log_dir, sess.graph)

    batch_size = config.batch_size
    no_of_batches = int(len(X_train)/batch_size)
    
    print("Starting training...")
    for i in tqdm(range(config.nb_epoch)):

        # Shuffle X_train, y_train
        random_permutation = np.random.permutation(len(X_train))
        X_train = X_train[random_permutation]
        y_train = y_train[random_permutation]

        for j in range(0,len(X_train)-batch_size,batch_size):
            # Get feed dict
            inp, out = X_train[j:j+batch_size], y_train[j:j+batch_size]
            inp = np.stack(inp,axis=0)
            out = np.stack(out,axis=0)

            # Forward pass & train step
            summary, __ = sess.run([Bob.merged, Bob.minimize],{Bob.input: inp, Bob.target: out})
            writer.add_summary(summary, i*no_of_batches+j/batch_size)
        
        # CV error
        cross_, accuracy_ = [], []
        if i % 2 == 0:
            for j in range(0,len(X_test)-batch_size,batch_size):
                # Get feed dict
                inp, out = X_test[j:j+batch_size], y_test[j:j+batch_size]
                inp = np.stack(inp,axis=0)
                out = np.stack(out,axis=0)
                # Run
                cross, accuracy = sess.run([Bob.cross_entropy, Bob.accuracy],{Bob.input: inp, Bob.target: out})
                cross_.append(cross)
                accuracy_.append(accuracy)
            # Print mean
            cross_, accuracy_ = np.asarray(cross_), np.asarray(accuracy_)
            print('\n Epoch',i,' CV cross entropy',np.mean(cross_))
            print(' Epoch',i,' CV accuracy',np.mean(accuracy_))

    print('Training completed !')
    saver.save(sess, "save/"+config.mode+"/actor.ckpt")
    print('Model saved.')

    # Print variables
    variables_names = [v.name for v in tf.global_variables() if 'Adam' not in v.name]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print("Variable: ", k, "Shape: ", v.shape)