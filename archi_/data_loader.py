import numpy as np
import pandas as pd
import os

from extract_sequence import extract_seq


# Keyboard size
num_roots = 24


def one_hot_melody(pitches,velocity,on_kick): # convert m to a dummy variable (a keyboard)
    output=np.zeros(num_roots+1+1) # +1 for silence, +1 for on_kick
    if velocity != 0:
        for note in pitches:
            output[int(note)]=1
        # rescale melody keyboard by velocity
        output = (output*velocity/100)
    else: # silence
        output[num_roots]=1
    # on_kick
    output[num_roots+1]=on_kick
    return output

def one_hot_chord(pitches,velocity): # convert c to a dummy variable (a keyboard)
    output=np.zeros(num_roots+1) # +1 for silence
    if velocity != 0:
        for note in pitches:
            output[int(note)]=1
    else: # silence
        output[num_roots]=1
    return output



class DataGenerator(object):

    def __init__(self, a_steps=8, b_steps=2):
        self.training_seq, self.target_seq = [], [] # X,y
        self.a_steps = a_steps # length of melody (listen)
        self.b_steps = b_steps # length of melody/harmony (attend & play)
        self.processed_tab = 0

    # tab format: Sequence of events (m_t, c_t)
    def process_tab(self,seq_m,seq_c,num, is_training=True):

        # Melody, Harmony lines
        melody_line, chord_line = [], []

        # process tab
        for m_t,c_t in zip(seq_m, seq_c):

            # pitch and velocity
            m_t_note, m_t_velocity, on_kick = m_t[0], m_t[1], m_t[2]
            c_t_note, c_t_velocity = c_t[0], c_t[1]

            if m_t_velocity != -1: # just a convenience for extract_seq (first token)

                # convert pitches to keyboard (one_hot)
                melody_line.append(one_hot_melody(m_t_note, m_t_velocity, on_kick))
                chord_line.append(one_hot_chord(c_t_note, c_t_velocity))

        self.processed_tab += 1
        # slice tab into smaller sequences 
        for i in range(0, len(melody_line)-(self.a_steps+self.b_steps), num): # always start on a kick
            if is_training == True:
                # only consider sequences where harmonic line plays once+
                notes_on = 0
                for keyboard in chord_line[i+self.a_steps:i+self.a_steps+self.b_steps]:
                    notes_on += np.count_nonzero(keyboard[:-1])
                # if harmonic line plays once+ during b_steps
                if notes_on > 2:           
                    # only consider sequences where melodic line plays once+
                    notes_on = 0
                    for keyboard in melody_line[i:i+self.a_steps+self.b_steps]:
                        notes_on += np.count_nonzero(keyboard[:-1])
                    # if melodic line plays once+ during a_steps + b_steps
                    if notes_on > 0:                                                                                                                ###############################################################
                        self.training_seq.append(np.asarray(melody_line[i:i+self.a_steps+self.b_steps]))    # 0,1,2,3 4,5,6,7
                        self.target_seq.append(np.asarray(chord_line[i+self.a_steps:i+self.a_steps+self.b_steps]))       # 6,7 # don't overfit..
            
            else:
                # consider all sequences
                self.training_seq.append(np.asarray(melody_line[i:i+self.a_steps+self.b_steps]))    # 0,1,2,3 4,5,6,7
                self.target_seq.append(np.asarray(chord_line[i+self.a_steps:i+self.a_steps+self.b_steps]))       # 6,7

    # process data files (.mid)
    def process_files(self,folder='train',mode='major'):
        print('Processing tabs in',mode,'key, from folder',folder,'...')

        titles = []
        for element in os.listdir(folder):
            if element.endswith('.mid'):
                titles.append(element[:-4])

        for title in titles:
            # midi to csv
            os.system('midicsv '+folder+'/'+title+".mid "+'output/tmp/'+title+".csv")

            # load .csv as pandas.Dataframe
            my_cols = ["track", "time", "info", "channel", "note", "velocity", "last_ts"]
            df = pd.read_csv("output/tmp/"+title+".csv", sep=',', names=my_cols, engine='c')

            try:
                # retrieve mode from pd.Dataframe
                mode_ = df.loc[df['info']==' Key_signature']['note'].values[-1] # major or minor
                mode_ = mode_[2:-1]
            except:
                print('\n',title,'Invalid mode ! Process aborted.')
                continue

            if mode_!=mode:
                print('\n',title,'is in',mode_,'key ! Process aborted.')
            else:
                print('\n Processing',title,'...')
                M,C,num,beat,first_kick,tonic_offset = extract_seq(df, melody_track=2, harmony_track=3)

                if beat != 'fail':
                    self.process_tab(M,C,num)

        print('\n Processed',self.processed_tab,'tabs in',mode,'key...')
        print('Issued',len(self.training_seq),'sequences of length',self.a_steps,'/',self.b_steps)
        #np.save(mode+"_"+folder+"_X.npy",self.training_seq)     # save X,y as .npy if running on Helios or whatever 
        #np.save(mode+"_"+folder+"_y.npy",self.target_seq)       # (don't need to repreprocess all the data)



if __name__ == "__main__":
    dataset = DataGenerator(a_steps=24, b_steps=24)
    dataset.process_files(folder='test',mode='minor')
    #print(dataset.training_seq[0],dataset.target_seq[0])