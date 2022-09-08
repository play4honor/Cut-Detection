#Probably the worst thing in here is my repeated use of fileName
# often for different things. Sorry!

import numpy as np
import subprocess
import datetime
import pandas
import sys
import os
import re

def load_cut_DF(fileName, clipType='all'):
    """Given a filename, loads the associated CSV with each play's cut data.
    Creates a unique filename for each play using the quarter and time (adds
    a 'b' if necessary to differentiate plays that have the same Q and T,
    removes semi-colons so filename is acceptable). Selects the appropriate 
    start and stop time for the clip based on clipType arg and adds them in 
    columns ['start'] and ['end_time']. Calculates the clip length based on 
    these columns. Prints and returns the dataframe."""

    df = pandas.read_csv(fileName)

    df = df[df['video'].isnull() == False].reset_index(drop=True)
    df['filename'] = 'Q'+df['q'].astype(str)+'_'+df['time'].astype(str)
    df['repeat'] = (df['filename'].duplicated())
    
    while df['repeat'].sum() > 0:
        df['filename'] = np.where(df['repeat'], df['filename']+'b', df['filename'])
        df['repeat'] = (df['filename'].duplicated())

    df['filename'] = df['filename'].str.replace(':', '')

    if clipType == 'a22':
        df['start'] = df['a22start']
        df['end_time'] = df['ezStart']
    elif clipType == 'ez':
        df['start'] = df['ezStart']
        df['end_time'] = df['a22start'].shift(-1)
    else:
        df['start'] = df['a22start']
        df['end_time'] = df['a22start'].shift(-1)
    
    df['length'] = df['end_time']-df['start']
    print (df)
    #sys.exit()
    return df

def convert_Seconds_To_HMS(time):
    """Given a time in HH:MM:SS format or similar, returns a time 
    in HH:MM:SS.XXX format"""

    #this is a sloppy hack that worked, so I stopped making it better
    #almost certainly a better way to do this

    temp = str(datetime.timedelta(milliseconds=1000*time))
    if len(temp) == 14:
        return temp.zfill(15)[0:12]
    else:
        return temp.zfill(8)+'.000'
    
def convert_Number_To_Down(number):
    """Given an input, returns the appropriate ordinal. Inputs should
    correspond to downs (1-4). Some plays (e.g. kickoffs) have no down
    and are instead 'NA' (so we return 'NA'). Elsewhere, our code 
    defines 2PA as down 0, so we return '2PT' for input 0."""

    #data should be good enough that there will never be an input that is not
    #a key in this dictionary, but it's easy enough to write error handling 
    #for the case when there is

    downDict = {0:'2PT',
                1:'1st', 
                2:'2nd', 
                3:'3rd', 
                4:'4th', 
                'NA':'NA'}
    
    return downDict[number]

def create_captions(df, fileName, team):
    """Given various inputs (a dataframe of plays, a filename, and a team 
    name), creates captions for each play, writing them to a local file. 
    Returns the caption file name for later use."""

    captionFileName = f"{fileName}_captions.srt"

    with open(captionFileName, 'w', encoding='utf-8') as file:
        #we create a caption for each row (play) in the dataframe
        for row in df.iterrows():
            index = row[0]+1
            
            #captions were weirdly drifting, so we adjust them to appear 3 
            # frames later for each clip spliced
            #this was another hack that seems to work for now
            start = row[1]['captionStart']+0.5+index/10
            
            captionStart = convert_Seconds_To_HMS(start)
            length = row[1]['length']
            
            #captions last either 4.5 seconds or until the end of the clip
            captionLength = min(4.5, length)
            
            captionEnd = convert_Seconds_To_HMS(start+captionLength)
            
            #srt files require a very specific format (mostly the header)
            header = f'{captionStart} --> {captionEnd}'
            qt = f"(Q{row[1]['q_x']},{row[1]['time_x']})"
            down = convert_Number_To_Down(row[1]['down'])
            togo = row[1]['togo']
            if togo == 0:
                togo = '2PT'
            los = row[1]['los']
            score = f"{team}:{row[1][team]} Opp:{row[1]['Opp']}"
            caption = (f'{index}\n{header}\n{qt}\n{down}&{togo}'
                       f' - {los}\n{score}\n\n')
            file.write(caption)

    return captionFileName

def select_plays(cutDataDF, fileName, min=0, max=None):
    """Given a dataframe of cut data and a file name, loads and prints a pbp 
    file with that file name, then inner joins the two dataframes on [date] 
    and [playid] columns (the result is a pbp dataframe that has exact cut 
    times for a22 and ez views of each play). Adds a column [captionStart]
    with values based on the length of the previous clip (and 0 for the first
    row. Returns the joined dataframe."""

    plays = pandas.read_csv(fileName)
    print (plays)

    joined = pandas.merge(cutDataDF,plays,how='inner',on=('date','playid'))

    joined['captionStart'] = joined['length'].shift(1).cumsum()
    joined['captionStart'] = joined['captionStart'].fillna(0)

    return joined

def ffmpeg_make_clips(df, file, fileName='test', combine=True):
    """Given a pbp dataframe with timings, a video file, and a file name,
    creates clips of the video file based on the timings in the dataframe
    using ffmpeg (1 for each row/play in the dataframe). Names the clips
    based on the Q,T naming convention in the dataframe (see load_cut_df
    for more) and the fileName arg provided to the function. Also writes 
    the clip's file name to a text file to allow it to be combined. Then
    uses the text file to combine the clips into a single video unless the
    combine Boolean arg is False."""

    ffmpegStrings = []
    string = ''

    for row in df.iterrows():
        length = row[1]['length']
        
        if np.isnan(length):
            end = ''
        
        else:
            end = f' -t {row[1]["length"]}'
        
        clipName = f'ffmpeg/{fileName}_{row[1]["filename"]}.mp4'
        ffmpegStrings.append(f'ffmpeg -hide_banner -ss {row[1]["start"]} -i {file} -codec copy{end} {clipName}')
        string += f"file '{clipName}'\n"
    
    #writing the clip names to a file
    with open(f'{fileName}_list.txt', 'w', encoding='utf-8') as file:
        file.write(string)
    
    #creating clips
    for string in ffmpegStrings:
        print (string)
        temp = string.split(' ')
        subprocess.run(temp)
    
    #combining created clips together
    if combine:
        combiner = f'ffmpeg -f concat -i {fileName}_list.txt -c copy {fileName}.mp4'
        print (combiner)
        subprocess.run(combiner.split(' '))

def find_videos(game):
    """Searching the videos folder in the current directory for any videos
    that have the specified game in the title. Prints each match, then 
    returns the name of the video, dropping .mp4 if it's there."""

    #a future update could try to make this work for other filetypes

    videos = [v.replace('.mp4', '') for v in os.listdir('videos')
                if game in v]
    for video in videos:
        print (video)
    return videos

if __name__ == '__main__':
    try:
        team = sys.argv[1]
        game = sys.argv[2]
        fileName = sys.argv[3]
        clipType = sys.argv[4]
        combineOnly = sys.argv[5]
    except: 
        print (f'1st arg is team abbreviation\n'
               f'2nd arg is game\n'
               f'3rd arg is fileName\n'
               f'4th arg is clip type (a22/ez/all)\n'
               f'5th arg is combine only (yes/no)\n\n'
               f'You entered:')
        print ([f'{x}' for x in sys.argv[1:]])

        sys.exit()

    ffmpeg_string = ''
    videos = find_videos(game)
    for game in videos:    
        
        #can skip making clips if they already exist using the combineOnly arg
        #this is vestigial, may remove soon
        if  combineOnly.lower() == 'yes':
            ffmpeg_string = (f'ffmpeg -f concat -i '
                             f'{game}_{fileName}_list.txt -c copy '
                             f'{game}_{fileName}.mp4')
            print (ffmpeg_string)
            input (".")
            subprocess.run(ffmpeg_string.split(' '))

        else:
            df = load_cut_DF(f'cutData/{game}.csv', clipType)
            print (df)
            
            plays = select_plays(df, f'cutData/{fileName}.csv')
            
            if plays.empty:
                print (f"No plays found for {game} and {fileName}")
                sys.exit()

            if not plays.empty:
                print (plays)
                videoName = f"{game}_{fileName}_{clipType}"
                captionFileName = create_captions(plays, videoName, team)
                ffmpeg_make_clips(plays, f'videos/{game}.mp4', videoName)

        #I don't know what this is for anymore
        #ffmpeg_string += f"file '{game}_{fileName}_{clipType}.mp4')"
        
        add_caption_string = (f'ffmpeg -i {videoName}.mp4 -i '
                              f'{captionFileName} -c copy -c:s '
                              f'mov_text {videoName}_subtitled.mp4')
        print (add_caption_string)
        subprocess.run(add_caption_string.split(' '))

#ffmpeg -f concat -safe 0 -i {fileName}.txt -c copy output.wav