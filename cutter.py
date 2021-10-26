import cv2
import numpy as np

import subprocess
import datetime
import pandas
import sys
import os

def load_cut_DF(fileName, fps, total_frames):
    df = pandas.read_csv(fileName)

    df = df[df['video'].isnull() == False].reset_index(drop=True)
    df['fps'] = fps
    df['filename'] = 'Q'+df['q'].astype(str)+' '+df['time'].astype(str)
    df = df[df['video'] != 'no film']
    df['repeat'] = (df['filename'].duplicated())
    
    while df['repeat'].sum() > 0:
        df['filename'] = np.where(df['repeat'], df['filename']+'b', df['filename'])
        df['repeat'] = (df['filename'].duplicated())

    df['start_frame'] = df['video'].apply(time_convert)*fps
    df['end_frame'] = df['start_frame'].shift(-1)
    df['end_time'] = df['video'].shift(-1)
    df['end_frame'] = np.where(np.isnan(df['end_frame']), total_frames, df['end_frame'])

    df['frames'] = df['end_frame'] - df['start_frame']
    df['length'] = df['frames'] / fps

    #df = df[df['frames']<fps*60]
    return df

def group_clip_frames(vc, fps, df, cutoff=None):
    
    curr_frame = None
    prev_frame = None

    idx = 0
    if not cutoff:
        cutoff = vc.get(cv2.CAP_PROP_FRAME_COUNT)

    startFrame = df.iloc[idx]['start_frame']
    vc.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fileName = df.iloc[idx]['filename'].replace(':', '')
    fileName = 'clips/'+fileName + '.mp4'
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v,')
    writer = cv2.VideoWriter(fileName, fourcc, fps, (width, height))
    
    while vc.isOpened():

        final_frame = int(df.iloc[idx]['end_frame']-1)
        ret, frame = vc.read()
        current_frame_number = vc.get(cv2.CAP_PROP_POS_FRAMES)
        
                
        if current_frame_number > cutoff:
            return 
        
        if ret:
            if curr_frame is None:
                curr_frame = frame
            else:
                prev_frame = curr_frame
                curr_frame = frame
            
            writer.write(frame)
                
            if current_frame_number > final_frame:
                idx += 1
                try:
                    fileName = df.iloc[idx]['filename'].replace(':', '')
                    fileName = 'clips/'+fileName + '.mp4'
                    print (fileName)
                    writer = cv2.VideoWriter(fileName, fourcc, fps, (width, height))
                except IndexError:
                    return 

        else:
            vc.release()
            break

    return fps

def detect_black_screen(frame):
    pass

def find_frames(vc, fps, df, fileName):
    print (df)

    idx = 0
    curr_frame = None
    prev_frame = None
    
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = vc.get(cv2.CAP_PROP_FRAME_COUNT)

    fileName = 'clips/'+fileName + '.mp4'
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v,')
    writer = cv2.VideoWriter(fileName, fourcc, fps, (width, height))
    
    while vc.isOpened():
    
        current_frame_number = vc.get(cv2.CAP_PROP_POS_FRAMES)
        startFrame = df.iloc[idx]['start_frame']
        
        if startFrame > current_frame_number:
            vc.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
        
        endFrame = df.iloc[idx]['end_frame']
        
        ret, frame = vc.read()
        
        if ret:
            if curr_frame is None:
                curr_frame = frame
            else:
                prev_frame = curr_frame
                curr_frame = frame

        if current_frame_number > endFrame or current_frame_number == total_frames:
            idx += 1
            try:
                print (df.iloc[idx]['playid'])
            except IndexError:
                vc.release()
                return
            continue
            
        else:        
            writer.write(frame)

def time_convert(x):
    try:
        h,m,s = map(int,x.split(':'))
    except ValueError:
        h,m,s = (np.nan,np.nan,np.nan)
    return (h*60+m)*60+s

def select_plays(df, fileName='', min=0, max=None):
    y = pandas.read_csv(fileName)
    joined = pandas.merge(df,y,how='inner', on='playid')
    joined = joined[joined['start_frame'] > min*fps]
    
    if max:
        joined = joined[joined['end_frame'] < max*fps]
        
    return joined

def write_ffmpeg_strings(df, file, fileName='test'):
    ffmpegStrings = []
    for row in df.iterrows():
        ffmpegStrings.append(f'ffmpeg -hide_banner -ss {row[1]["video"]} -i {file} -codec copy -t {row[1]["length"]} ffmpeg/{fileName}{row[0]}.mp4')
    for string in ffmpegStrings:
        print (string)
        temp = string.split(' ')
        subprocess.run(temp)

if __name__ == '__main__':
    
    #game = '2021.09.13_Raiders_Ravens'
    #game = '2021.09.19_Ravens_Chiefs'
    #game = '2021.09.26_Lions_Ravens'
    game = '2021.10.03_Broncos_Ravens'
    vc = cv2.VideoCapture(f'videos/{game}.mp4')
    fps = int(vc.get(cv2.CAP_PROP_FPS))
    print (fps)
    
    
    videos = os.listdir('ffmpeg')
    for video in videos:
        if 'bowser' in video:
            print (video)
    
    with open('bowserPass.txt', 'w', encoding='utf-8') as file:
        for video in videos:
            file.write(f"file '{video}'\n")
        
    total_frames = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    
    #df = load_cut_DF(f'cutData/{game}.csv', 60, total_frames)
    df = load_cut_DF(f'cutData/{game}.csv', 30, total_frames)
    
    #columns = ['start_frame', 'end_frame', 'frames', 'type']
    #df['type'] = 'a22'
    #df[columns].to_csv('BALDENcuts.csv', index=False)
    #sys.exit()
    
    #fileName = 'pass DEN'
    #fileName = 'pass DET'
    #fileName = 'pass KC'
    #fileName = 'pass LV'

    fileName = 'bowser broncos run'
    fileName = 'bowser broncos dropback'
    plays = select_plays(df, f'cutData/{fileName}.csv')
    
    #write_ffmpeg_strings(plays, f'videos/{game}.mp4', f"{game}_bowser_pass")
    #find_frames(vc, 30, plays, f'{fileName}')
    #find_frames(vc, fps, plays, 'stephens raiders.csv')
    #find_frames(vc, fps, df, 'cleveland chiefs.csv')
    
    #group_clip_frames(vc, fps, df)


    #fileName = 'queen run 2021-09-13' 
    #fileName = 'queen run 2021-09-19' 
    #fileName = 'queen run 2021-09-26' 
    #fileName = 'queen db 2021-09-13' 
    #fileName = 'queen db 2021-09-19' 
    #fileName = 'queen db 2021-09-26' 
    #fileName = 'Ravens Broncos D3'

#ffmpeg -hide_banner -i "videos/2021.10.03_Broncos_Ravens.mp4" -filter:v "select='gt(scene,0.2)',showinfo" -f null - 2>&1> cutDetectTest.txt
#ffmpeg -hide_banner -i "videos/2021.10.03_Dolphins_Colts.mp4" -filter:v "select='gt(scene,0.2)',showinfo" -f null - 2> coltsDolphinsCuts.txt

