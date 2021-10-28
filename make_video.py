import subprocess
import pandas
import sys
import os
import re
import numpy as np

def time_convert(x):
    try:
        h,m,s = map(int,x.split(':'))
    except ValueError:
        h,m,s = (np.nan,np.nan,np.nan)
    return (h*60+m)*60+s

def load_cut_DF(fileName):
    df = pandas.read_csv(fileName)

    df = df[df['video'].isnull() == False].reset_index(drop=True)
    df['filename'] = 'Q'+df['q'].astype(str)+'_'+df['time'].astype(str)
    df['repeat'] = (df['filename'].duplicated())
    
    while df['repeat'].sum() > 0:
        df['filename'] = np.where(df['repeat'], df['filename']+'b', df['filename'])
        df['repeat'] = (df['filename'].duplicated())

    df['end_time'] = df['secs'].shift(-1)
    df['length'] = df['end_time']-df['secs']
    df['filename'] = df['filename'].str.replace(':', '')
    return df

def select_plays(df, fileName='', min=0, max=None):
    y = pandas.read_csv(fileName)
    joined = pandas.merge(df,y,how='inner', on=('date','playid'))
    return joined

def ffmpeg_make_clips(df, file, fileName='test', combine=True):
    ffmpegStrings = []
    string = ''
    for row in df.iterrows():
        clipName = f'ffmpeg/{fileName}_{row[1]["filename"]}.mp4'
        ffmpegStrings.append(f'ffmpeg -hide_banner -ss {row[1]["secs"]} -i {file} -codec copy -t {row[1]["length"]} {clipName}')
        #ffmpegStrings.append(f'ffmpeg -hide_banner -ss {row[1]["secs"]} -i {file} -codec copy -t {row[1]["length"]} ffmpeg/{fileName}{row[0]}.mp4')
        string += f"file '{clipName}'\n"
    with open(f'{fileName}_list.txt', 'w', encoding='utf-8') as file:
        file.write(string)
    for string in ffmpegStrings:
        print (string)
        temp = string.split(' ')
        subprocess.run(temp)
    if combine:
        combiner = f'ffmpeg -f concat -i {fileName}_list.txt -c copy {fileName}.mp4'
        print (combiner)
        subprocess.run(combiner.split(' '))

if __name__ == '__main__':
    try:
        game = sys.argv[1]
        fileName = sys.argv[2]
        combine = sys.argv[3]
    except: 
        print ("1st arg is game\n2nd arg is fileName\n3rd arg is combine boolean")
        sys.exit()
    df = load_cut_DF(f'cutData/{game}.csv')
    print (df)
    plays = select_plays(df, f'cutData/{fileName}.csv')
    print (plays)
    #sys.exit()
    ffmpeg_make_clips(plays, f'videos/{game}.mp4', f"{game}_{fileName}", combine)

#ffmpeg -f concat -safe 0 -i {fileName}.txt -c copy output.wav