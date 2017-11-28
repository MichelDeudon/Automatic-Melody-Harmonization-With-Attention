import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings("ignore")


from actor import Actor
from config import get_config
import data_loader
from extract_sequence import extract_seq



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
### Harmonize ###
#################

total_accuracy, total_cross_entropy = [], []

def add_chords(title, actor, melody_track=2, harmony_track=3, mode='major'):

    # Load .csv as pandas.Dataframe
    my_cols = ["track", "time", "info", "channel", "note", "velocity", "last_ts"]
    df=pd.read_csv("output/tmp/"+title+".csv", sep=',', names=my_cols, engine='c')

    try:
        mode_ = df.loc[df['info']==' Key_signature']['note'].values[0] # major or minor
        mode_ = mode_[2:-1]
    except:
        print('\n',title,'invalid mode ! Process aborted.')
        return 0

    if mode_ != mode:
        # print('\n',title,'is in',mode_,'key ! Process aborted.')
        return 0

    print("\nAdding chords to: ",title,"played in",mode_,"key")
    # Extract Melody and Harmony from .csv (time/frequency analysis)
    M,C,num,beat,first_kick,tonic_offset = extract_seq(df, melody_track=melody_track, harmony_track=harmony_track)
    if beat == 'fail':
        print('\n',title,'invalid beat ! Process aborted.')
        return 0

    # Create data generator
    dataset = data_loader.DataGenerator(a_steps=config.a_steps,b_steps=config.b_steps)
    dataset.process_tab(M, C, num, is_training=False)
    X_test, y_test = dataset.training_seq, dataset.target_seq # (0,1,2...,6,7) --> (4,5,6,7) --> (4)

    # Harmonize Melody
    target, prediction, accuracies, cross_entropies = [], [], [], []
    #print("Decoding...")
    for j in range(0,len(X_test)-config.batch_size, config.batch_size):
        # Get feed dict
        inp, out = X_test[j:j+config.batch_size], y_test[j:j+config.batch_size]
        inp = np.stack(inp,axis=0)
        out = np.stack(out,axis=0)

        # Forward pass
        pointr, cross, accuracy, target_chords, played_chords = sess.run([actor.pointing_, actor.cross_entropy, actor.accuracy, actor.target_chords, actor.played_chords],{actor.input: inp, actor.target: out})
        accuracies.append(accuracy)
        cross_entropies.append(cross)

        # Build Harmony line
        for _ in range(config.batch_size): #################################################
            for tps in range(num):
                target.append(target_chords[_][tps])
                prediction.append(played_chords[_][tps])
            #visualize_attention(played_chords[_])

    try:
        target = np.stack(target,0)
        target = target[:,:-1]
        #visualize_attention(target[:100])
    except:
        pass

    try:
        prediction = np.stack(prediction,0)
        prediction = prediction[:,:-1]
        #visualize_attention(prediction[:100])
        print("Decoding completed!",'(predicted',len(prediction),'chords)')
    except:
        print('\n Prediction empty ! Process aborted.')
        return 0

    # Compare to ground truth
    accuracies, cross_entropies = np.asarray(accuracies), np.asarray(cross_entropies)
    print('Accuracy:',np.mean(accuracies))
    print('Cross entropy:',np.mean(cross_entropies))
    total_accuracy.append(np.mean(accuracies))
    total_cross_entropy.append(np.mean(cross_entropies))

    

    # Clean .csv Header
    header_df=df.loc[df["track"]<=1].loc[df["info"]!=" End_of_file"]
    header_df=header_df.replace(' "major"','major')
    header_df=header_df.replace(' "minor"','minor')
    header_df = header_df.loc[header_df["info"]!= ' SMPTE_offset']
    header_df = header_df.loc[header_df["info"]!= ' System_exclusive']
    header_df = header_df.loc[header_df["info"]!= ' Sequencer_specific']

    # Prepare .csv Melody line
    before_melody=[]
    before_melody.append([melody_track, 0, 'Start_track'])
    before_melody.append([melody_track, 0, 'MIDI_port', 0])
    before_melody.append([melody_track, 0, 'Title_t', "Melody"])
    before_melody.append([melody_track, 0, 'Program_c', 1, 19,'',''])
    before_melody_df=pd.DataFrame(before_melody, columns=my_cols)

    # Clean .csv Melody line
    dfM = df.loc[df["track"]==melody_track]
    dfM = dfM.loc[(dfM["info"] == " Note_on_c")+(dfM["info"] == " Note_off_c")]
    dfM.fillna(0,inplace=True)

    # Transit .csv Melody / Harmony lines
    milieu=[]
    milieu.append(([melody_track, dfM["time"].values.max(), "End_track","","","",""]))
    milieu.append(([3, 0, "Start_track","","","",""]))
    milieu.append([3, 0, 'MIDI_port', 0])
    milieu.append([3, 0, 'Title_t', "Chords"])
    milieu.append([3, 0, 'Program_c', 1, 19])
    milieu_df=pd.DataFrame(milieu,columns=my_cols)

    # Build .csv Harmony line
    octave=int(dfM["note"].values.astype(int)[0]/12)-1
    data_chords=[]
    
    for k in range(len(prediction)):
        t = first_kick + (k+config.a_steps)*beat
        key=np.where(prediction[k]!=0)[0]

        for note in key:
            n = tonic_offset+note+octave*12
            data_chords.append(([3,t,'Note_on_c',2,n,80,""]))
            data_chords.append(([3,t+2*beat,'Note_off_c',2,n,80,""]))
            
    chords_line=pd.DataFrame(data_chords, columns=my_cols).sort_values(by=["time","info"],ascending=[True,False])

    # Close .csv Harmony line
    end = []
    end.append([3,dfM["time"].values.max(),'End_track',"","","",""])
    end.append([0,0,'End_of_file',"","","",""])
    end_df = pd.DataFrame(end,columns=my_cols)

    # Concatenate .csv
    dfSortie=pd.concat([header_df, before_melody_df, dfM, milieu_df, chords_line, end_df])
    dfSortie.to_csv("output/tmp/harmonized_"+title+".csv",header=False,index=False)
    #print(dfSortie)

    # Save midi
    os.system('csvmidi '+'output/tmp/harmonized_'+title+".csv "+'output/harmonized_'+title+'.mid')
    print("Saved midi file")
    return np.mean(accuracies), np.mean(cross_entropies)




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
        print("Model restored from",config.mode)

    # Collect midi songs in correct mode
    print("Playing in",config.mode,'key, from folder',config.test_folder,'...')
    titles = []
    for element in os.listdir(config.test_folder):
        if element.endswith('.mid'):
            titles.append(element[:-4])

    # Harmonize pieces
    for title in titles:
        # midi to csv
        os.system('midicsv '+config.test_folder+'/'+title+".mid "+'output/tmp/'+title+".csv")
        add_chords(title, Bob, melody_track=2, harmony_track=3, mode=config.mode)

    print('\n Mean Accuracy:',np.mean(np.asarray(total_accuracy)))
    print(' Mean Cross entropy:',np.mean(np.asarray(total_cross_entropy)))