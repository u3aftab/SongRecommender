import hdf5_getters 
#from https://github.com/tbertinmahieux/MSongsDB/blob/master/PythonSrc/hdf5_getters.py
import pandas as pd
import cPickle as cp
import os


### configs        
f_dir = '/Users/umaraftab/Downloads/MillionSongSubset/data/'

song_funs = {
'artist_id': hdf5_getters.get_artist_id,
'song_id': hdf5_getters.get_song_id,
'title': hdf5_getters.get_title,
'duration': hdf5_getters.get_duration,
'key': hdf5_getters.get_key,
'key_confidence': hdf5_getters.get_key_confidence,
'loudness': hdf5_getters.get_loudness,
'mode': hdf5_getters.get_mode,
'mode_confidence': hdf5_getters.get_mode_confidence,
'tempo': hdf5_getters.get_tempo}

artist_vars = {
'similar_artists': hdf5_getters.get_similar_artists,
'artist': hdf5_getters.get_artist_name}


### dataframe transform method
def h5_to_df(hdf5_file):
    
    df = []
    cols = song_funs.keys()
    
    for i in xrange(hdf5_getters.get_num_songs(hdf5_file)):
        row = []
        for col in cols:
            row.append(song_funs[col](hdf5_file, i))
        df.append(row)     
    
    return pd.DataFrame(df, columns=cols)


### process and format song data
# formatting statistics
file_count = 0
artists_vars_dict = {}
songs = pd.DataFrame()

for root, dirs, files in os.walk(f_dir):
    for f in files:
        if f[-3:] != '.h5':
            continue
        file_count += 1
        
        h5_file_path = os.path.join(root, f)
        h5_file = hdf5_getters.open_h5_file_read(h5_file_path)
        songs = songs.append(h5_to_df(h5_file))
        
        artist_id = hdf5_getters.get_artist_id(h5_file)
        if artist_id in artists_vars_dict:
            h5_file.close()
            continue

        artists_vars_dict[artist_id] = {}
        for var in artist_vars:
            artists_vars_dict[artist_id][var] = artist_vars[var](h5_file)
            
        h5_file.close()
        
cp.dump([artists_vars_dict, songs], open('song_data.pckl', 'wb'))

print 'total_songs:', len(songs)
print 'files_processed:', file_count
print 'number of artists in songs dataframe:', len(set(songs['artist_id']))
print 'number of artists in artists dictionary:', len(artists_vars_dict)
