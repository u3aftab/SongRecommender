import pandas as pd
import numpy as np
import operator
import cPickle as cp

class SongRecommender():
    """SongRecommender is a system to recommend a song to a user given previous 
    songs they've listened to.
    """
    
    def __init__(self, 
                 songs, 
                 artists_vars, 
                 weights = [3, 2, 1, 1, 1]):
        """Creates a recommendation engine given a repository of songs.
        
        Params:
        ------
        songs : un-indexed pandas dataframe with columns song_id, mode_confidence, 
        key_confidence, key, title, duration, loudness, artist_id, tempo, mode
        artists_vars : dictionary with the artist's name and similar artists, for a given
        artist_id
        weights : how much to weight key, mode, loudness, duration and tempo, respectively,
        for each song in the ranking
        
        Returns:
        ------
        Recommender method
        """
        self.songs = songs.set_index(['song_id'])
        self.artist_songs = songs.set_index(['artist_id'])
        self.artists_vars = artists_vars
        self.weights = weights
        self.random_song_id = [self.songs.sample()['artist_id'][0]]

    def get_similar_artists(self, song_list):
        """Returns a list of lists with artists similar to artists in the input-ed song list
        
        Params
        ------
        song_list : list of songs
        
        Returns
        -------
        freq_list : sorted list of lists, where a nested list closer to the front contains
        artists that are more similar to artists in the song_list
        
        """
        artist_freq_dict = {}
        for song in song_list:
            similar_artists = self.artists_vars[self.songs['artist_id'][song]]['similar_artists']
            for artist_i in similar_artists:
                if artist_i in artist_freq_dict:
                    artist_freq_dict[artist_i] += 1
                else:
                    artist_freq_dict[artist_i] = 1

        artist_freq_dict = sorted(artist_freq_dict.items(), 
                                  key=operator.itemgetter(1),
                                  reverse=True)
        freq_list = [[]]
        i, j = artist_freq_dict[0][1], 0
        while j < len(artist_freq_dict):
            if artist_freq_dict[j][1] != i:
                i = artist_freq_dict[j][1]
                freq_list.append([])
            freq_list[-1].append(artist_freq_dict[j][0])
            j += 1

        return freq_list
    
    def create_weight_dict(self,
                           var, 
                           song_list):
        """Returns weights for categorical variables
        
        Params
        ------
        var : categorical variable that is a feature of a song
        song_list : list of songs from which to train a recommender on
        
        Returns
        ------
        freq_dict  dictionary with frequency of the input-ed variable var
        """
        freq_dict = {}
        freq_dict['N'] = 1.0 * len(song_list) 
        for song in song_list:
            i = self.songs[var][song]
            if i in freq_dict:
                freq_dict[i] += self.songs[var+'_confidence'][i]
            else:
                freq_dict[i] = self.songs[var+'_confidence'][i]
        return freq_dict
    
    def get_row_score(self,
                      row_key,
                      row_mode,
                      row_loudness,
                      row_duration,
                      row_tempo,
                      key_d,
                      mode_d, 
                      loudness_mean, 
                      loudness_std, 
                      duration_mean, 
                      duration_std, 
                      tempo_mean, 
                      tempo_std):
        """Returns a song's, a row in a dataframe, match to the input-ed song list
        
        Params
        ------
        row_key, row_mode,row_loudness, row_duration, row_tempo : the row's features
        
        key_d, mode_d : dictionary with the input-ed song list's statistics
        
        loudness_mean, loudness_std, duration_mean, duration_std,  
        tempo_mean, tempo_std : the statistics of the continous feature values
        
        Returns
        ------
        the weighted row score
        """
        out = []
        
        out.append(key_d.get(row_key, 0)/key_d['N'])
        out.append(mode_d.get(row_mode, 0)/mode_d['N'])
        
        try:
            out.append(max((loudness_std - abs(loudness_mean - row_loudness))/loudness_std, 0))
        except ZeroDivisionError:
            out.append(0)
            
        try:
            out.append(max((duration_std - abs(duration_mean - row_duration))/duration_std, 0))
        except ZeroDivisionError:
            out.append(0)
            
        try:
            out.append(max((tempo_std - abs(tempo_mean - row_tempo))/tempo_std, 0))
        except ZeroDivisionError:
            out.append(0)
        
        return np.dot(out, self.weights)
        
        
    def recommend(self, 
                  song_list=[]):
        """Returns a song recommendation given a list of songs a user likes.  This will only 
        provide a recommendation for a song that was in the data the model was built on.
        
        
        Params
        ------
        song_list : list of song_ids for songs
        
        Returns
        ------
        A string that has the song_id, song name and artist name for a song
        that is recommended 
        """
        # empty song_list
        if len(song_list) == 0:
            rec = self.artist_songs.sample()
            return 'song_id: {}, {} by {}'.format(rec['song_id'][0],
                                                  rec['title'][0],
                                                  self.artists_vars[rec.index[0]]['artist'])
        
        # get similar artists
        similar_artists = SongRecommender.get_similar_artists(self, song_list)
        like_artists_songs = self.artist_songs.loc[similar_artists[0]+self.random_song_id].dropna()
        like_artists_songs = like_artists_songs[~like_artists_songs.isin(song_list)]
        i = 1
        while len(like_artists_songs) < 1 and i < len(similar_artists):
            like_artists_songs = self.artist_songs.loc[similar_artists[i]+self.random_song_id].dropna()
            like_artists_songs = like_artists_songs[~like_artists_songs.isin(song_list)]
            i += 1
        if len(like_artists_songs) < 1:
            like_artists_songs = self.artist_songs
            like_artists_songs = like_artists_songs[~like_artists_songs.isin(song_list)]
            
            
        # input songs statistics for categorical features
        # key signature and mode
        key_dict = SongRecommender.create_weight_dict(self, 'key', song_list)
        mode_dict = SongRecommender.create_weight_dict(self, 'mode', song_list)
        
        # statistics for continuous features
        # loudness, duration, tempo
        loudness = like_artists_songs['loudness']
        loudness_mean, loudness_std = np.mean(loudness), np.std(loudness)
        duration = like_artists_songs['duration']
        duration_mean, duration_std = np.mean(duration), np.std(duration)
        tempo = like_artists_songs['tempo']
        tempo_mean, tempo_std = np.mean(tempo), np.std(tempo)
        
        like_artists_songs['row_score'] = np.vectorize(SongRecommender.get_row_score)(self,like_artists_songs['key'],
                                           like_artists_songs['mode'],
                                           like_artists_songs['loudness'], 
                                           like_artists_songs['duration'], 
                                           like_artists_songs['tempo'],
                                           key_dict,
                                           mode_dict,
                                           loudness_mean, 
                                           loudness_std, 
                                           duration_mean, 
                                           duration_std, 
                                           tempo_mean, 
                                           tempo_std)
        
        rec = like_artists_songs.sort_values(['row_score'], ascending = False).dropna().head(1)
        
        if len(rec) < 1:
            rec = self.artist_songs.sample()
            return 'song_id: {}, {} by {}'.format(rec['song_id'][0],
                                                  rec['title'][0],
                                                  self.artists_vars[rec.index[0]]['artist'])
        
        return 'song_id: {}, {} by {}'.format(rec['song_id'][0],
                                              rec['title'][0],
                                              self.artists_vars[rec.index[0]]['artist'])
