import pandas as pd
import numpy as np
import os

from collections import Counter

# Keyboard size
num_roots = 24

# Map key signatures to tonic offset (same dict major/minor key. Ref A min)
# Sharps: C maj (0#), G maj(1#), D maj (2#), A maj (3#), E maj (4#), B maj (5#), F# maj (6#), C# maj (7#)
# Flats: C maj (0b), F maj(1b), Bb maj (2b), Eb maj (3b), Ab maj (4b), Db maj (5b), Gb maj (6b), Cb maj (7b)
key_to_tonic = {0:0, 1:7, 2:2, 3:9, 4:4, 5:11, 6:6, 7:1, -1:5, -2:10, -3:3, -4:8, -5:1, -6:6, -7:11}


def extract_seq(df, melody_track=2, harmony_track=3):

    # collect melody data
    melody = df.loc[df['track']==melody_track]
    melody_on = melody.loc[melody['info']==' Note_on_c']
    melody_on = melody_on.loc[melody_on['velocity']!=0]
    melody_on = melody_on[["time", "note", "velocity"]]
    melody_on = melody_on.astype(int)
    #print(melody[:10])
    #melody_off = melody.loc[melody['info']==' Note_off_c']
    #melody_off = melody_off[["time", "note"]]
    #melody_off = melody_off.astype(int)

    # collect harmony data
    harmony = df.loc[df['track']==harmony_track]
    harmony_on = harmony.loc[harmony['info']==' Note_on_c']
    harmony_on = harmony_on.loc[harmony_on['velocity']!=0]
    harmony_on = harmony_on[["time", "note", "velocity"]]
    harmony_on = harmony_on.astype(int)
    #print(harmony[:10])
    #harmony_off = harmony.loc[harmony['info']==' Note_off_c']
    #harmony_off = harmony_off[["time", "note"]]
    #harmony_off = harmony_off.astype(int)

    # sanity check: melody not empty
    try:
        times = np.unique(melody_on['time'].values)
        last_t = times[-1]
    except:
        print('Melody is empty ! Process aborted.')
        return 'fail', 'fail', 'fail', 'fail', 'fail', 'fail'

    # collect key signature for transposition
    try:
        # from MIDI key meta data
        key = int(df.loc[df['info']==' Key_signature']['channel'].values[-1]) # Key signature is 0 for the key of C, a positive value for each sharp above C, or a negative value for each flat below C, thus in the inclusive range −7 to 7 
        tonic_offset = key_to_tonic[key] # offset to transpose the piece to C maj (if major scale) or A min (if minor scale)
        print('Tonic:',tonic_offset)
    except:
        print('Implement Krumhansl-Schmuckler key id. algorithm (TODO) ! Process aborted.')
        return 'fail', 'fail', 'fail', 'fail', 'fail', 'fail'

    # collect beat from temporal analysis
    division = float(df[:1]['velocity'].values[0]) # the number of clock pulses per quarter note (typically 480)
    df_header = df.loc[df['track']==1]
    try:
        # from MIDI time meta data
        time_signature = df_header.loc[df_header['info']==' Time_signature'] # Num / Denom. Denom specifies the denominator as a negative power of two, for example 2 for a quarter note, 3 for an eighth note, etc. 
        num, denom = int(time_signature['channel'].values[0]), 2**(-int(time_signature['note'].values[0]))
        print('Time signature:',num,int(1/denom))
    except:
        print('No time signature ! Process aborted.')
        return 'fail', 'fail', 'fail', 'fail', 'fail', 'fail'

    # num, denom = 2*num, denom/2 # for higher resolution
    beat = int(division*denom/0.25) # beat = quarter_note_pulse * ratio_to_quarter_note

    first_kick = 0 # times[0]
    window = beat/10 # window for harmonization (don't constrain heard harmonization on beats)
    t = first_kick

    M, C = [(np.rint([-1]),-1,-1)], [(np.rint([-1]),-1)]   # this is useless (just helps for save npy)
    count_silence_m, count_silence_c, count_ = 0.0, 0.0, 0.0
    while t < times[-1]:

        # track measure (on_kick ft.)
        if t%(num*beat)==0:
            on_kick = 1
        else:
            on_kick = 0

        # melodic line
        m_t = melody_on.loc[(melody_on['time']<(t+1))*(melody_on['time']>=(t))]
        m_t_note, m_t_velocity = m_t["note"].values, m_t["velocity"].values
        # if silence
        if len(m_t_note)==0: 
            M.append(("silence",0,on_kick))
            count_silence_m+=1.0
        # else tranpose
        else:
            m_t_note = (np.rint(m_t_note) - tonic_offset) % num_roots
            M.append((m_t_note,np.mean(m_t_velocity),on_kick))

        # harmonic line
        c_t = harmony_on.loc[(harmony_on['time']<(t+1))*(harmony_on['time']>=(t-window))]
        c_t_note, c_t_velocity = c_t["note"].values, c_t["velocity"].values
        # if silence
        if len(c_t_note)==0: 
            C.append(("silence",0))
            count_silence_c+=1.0
        # else tranpose
        else:
            c_t_note = (np.rint(c_t_note) - tonic_offset) % num_roots
            C.append((c_t_note,np.mean(c_t_velocity)))

        count_+=1.0
        t += beat

    print('Melody silence ratio:',count_silence_m/count_)
    print('Harmony silence ratio:',count_silence_c/count_)
    return M, C, num, beat, first_kick, tonic_offset




if __name__ == "__main__":

    titles = []
    for element in os.listdir('csv'):
        if element.endswith('.csv'):
            titles.append(element)

    for title in titles:
        my_cols = ["track", "time", "info", "channel", "note", "velocity", "last_ts"]
        df=pd.read_csv("csv/"+title, sep=',', names=my_cols, engine='c')
        mode = df.loc[df['info']==' Key_signature']['note'].values[0] # major or minor
        mode = mode[2:-1]
        print('\n',mode, title[:-4])

        M, C, num, beat, first_kick, tonic_offset = extract_seq(df)

        if beat != 'fail':
            #np.save("clean_data/"+mode+'/'+title[:-4]+"_melody.npy",M)
            #np.save("clean_data/"+mode+'/'+title[:-4]+"_chords.npy",C)
            pass