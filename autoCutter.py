import subprocess
import pandas
import sys
import os
import re

def determine_frame_rate(fileName):
    if 'mp4' in fileName:
        ffmpegCommand = (f'ffmpeg -hide_banner -i videos/{fileName}')
    else:
        ffmpegCommand = (f'ffmpeg -hide_banner -i videos/{fileName}.mp4')
    video_data = subprocess.run(ffmpegCommand.split(' '), 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.STDOUT, 
                                  encoding='utf-8')
    stdout = video_data.stdout

    pattern = '(fps, [0-9]*\.?[0-9]*)'
    try:
        fps = re.search(pattern, stdout).group(1).replace('fps, ', '')
    except AttributeError:
        print (stdout)
        return 
    return fps

def find_black_frames(fileName, threshold):
    
    ffmpegCommand = (f'ffmpeg -hide_banner -i videos/{fileName}.mp4 '
                    f'-vf blackdetect=d=0:pix_th={threshold} -f null -')
    print (ffmpegCommand)

    black_frames = subprocess.run(ffmpegCommand.split(' '), 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.STDOUT, 
                                  encoding='utf-8')
    #stderr = black_frames.stderr
    stdout = black_frames.stdout
    
    savePath = f'black_frames/{fileName}_black_frames_stdout.txt'
    
    with open(savePath, 'w', encoding='utf-8') as file:
            black_frames = file.write(stdout)
    
    return stdout

def parse_black_frame_text(fileName, fps):
    loadPath = f'black_frames/{fileName}_black_frames_stdout.txt'
    
    try:
        with open(loadPath, 'r', encoding='utf-8') as file:
    
            black_frames = file.read()
            print (f'Black frame data for {fileName} '
                   f'loaded from local data')
    
    except FileNotFoundError:
    
        print (f'No black frame data for {fileName} found, '
               f'checking for black frames now')
        black_frames = find_black_frames(fileName, 0.05)

    pattern = '(black_start:[0-9]*.[0-9]* black_end:[0-9]*.[0-9]* black_duration:[0-9]*.[0-9]*)'
    black_frame_timing = re.findall(pattern, black_frames)
    df = pandas.DataFrame()
    temp = {}
    for black_frame in black_frame_timing:
        for each in black_frame.split(' '):
            keyValue = each.split(':')
            temp[keyValue[0]] = float(keyValue[1])
        df = df.append(temp, ignore_index=True)
    df.to_csv(f'black_frames/{fileName}.csv', index=False)
    return df

def remove_black_frames(fileName):
    df = parse_black_frame_text(fileName, 30)
    print (df)
    ffmpegCommand = 'ffmpeg -hide_banner -ss 0 -to '
    suffix = ''
    sys.exit()
    for row in df.iterrows():
        index = row[0]
        first_black_frame = row[1]['black_start']
        final_black_frame = row[1]['black_end']
        
        ffmpegCommand += str(first_black_frame)
        ffmpegCommand += f' -i {fileName}.mp4 -ss {final_black_frame} -to '
        suffix += f' -map {index}:v -c copy {fileName}part{index}.ts'
    print (ffmpegCommand)
    ffmpegCommand = ffmpegCommand.rsplit(' -ss', maxsplit=1)[0]
    ffmpegCommand += suffix
    print (ffmpegCommand)
    
    subprocess.run(ffmpegCommand.split(' '))
        #ffmpeg -ss 0 -to 8.9 -i short.mp4 -ss 9.76667 -to 207.1 -i short.mp4 -map 0:v -c copy part1.ts -map 1:v -c copy part2.ts
        #ffmpeg -ss 0 -to 8.9 -i short.mp4 -ss 9.76667 -to 207.1 -i short.mp4 -ss 207.967 -to

#"ffmpeg -i {0}  -filter:v \"select='gt(scene,0.4)',showinfo\"  -f {1}  - 2> ffout".format(inputName, outputFile)

def find_scene_changes(fileName, threshold):
    ffmpegCommand = (f"ffmpeg -hide_banner -i videos/{fileName}.mp4 "
                    f'-filter:v select\=gt(scene\,{threshold})\,showinfo'
                    f" -f null -")
    ffmpegCommand = f"ffmpeg -hide_banner -i videos/{fileName}.mp4 -filter:v \"select='gt(scene,0.1)',showinfo\" -f null - 2>> scene_changes/{fileName}_scene_changes_stdout.txt"

    print (ffmpegCommand)
    print (ffmpegCommand.split(' '))
    scene_changes = subprocess.run(ffmpegCommand.split(' '),
                   stdout=subprocess.PIPE, 
                   stderr=subprocess.STDOUT, 
                   encoding='utf-8')

    stdout = scene_changes.stdout
    savePath = f'scene_changes/{fileName}_scene_changes_stdout.txt'
    print (stdout)
    #with open(savePath, 'w', encoding='utf-8') as file:
    #        scene_changes = file.write(stdout)
    
    return stdout

def parse_scene_change_text(fileName, fps):
    loadPath = f'scene_changes/{fileName}_scene_changes_stdout.txt'
    
    try:
        with open(loadPath, 'r', encoding='utf-8') as file:
    
            scene_changes = file.read()
            print (f'Scene change data for {fileName} '
                   f'loaded from local data')
    
    except FileNotFoundError:
    
        print (f'No scene change data for {fileName} found, '
               f'checking for scene changes now')
        scene_changes = find_scene_changes(fileName, 0.1)

    pattern = '(n:\s*[0-9]* pts:\s*[0-9]* pts_time:\s*[0-9]*\.[0-9]*)' 
    scene_change_timing = re.findall(pattern, scene_changes)
    df = pandas.DataFrame()
    temp = {}
    for scene_change in scene_change_timing:
        while ': ' in scene_change:
            scene_change = scene_change.replace(': ', ':')
        for each in scene_change.split(' '):
            keyValue = each.split(':')
            temp[keyValue[0]] = float(keyValue[1].replace(' ',''))
        df = df.append(temp, ignore_index=True)
    df.to_csv(f'scene_changes/{fileName}.csv', index=False)
    return df

def check_scene_change_times(fileName):
    df = pandas.read_csv(f'{fileName}_cuts.csv')
    print (df)
    for row in df.iterrows():
        string = f'ffmpeg -hide_banner -ss {row[1][0]} -i videos/{fileName}.mp4 -vframes 1 -q:v 2 output{row[0]}.jpg'
        print (string)
        subprocess.run(string.split(' '))

def cut_video(file):
    pass

#ffmpeg -hide_banner -i videos/2021.10.03_Broncos_Ravens.mp4 -filter:v "select='gt(scene,0.2)',showinfo" -f null 2>&1 >> cutDetectTest.txt
#ffmpeg -hide_banner -i videos/2021.10.24_Ravens_Bengals.mp4 -filter:v "select='gt(scene,0.1)',showinfo" -f null 2>> scene_changes/2021.10.24_Ravens_Bengals_scene_changes_stdout.txt

try: 
    videos = [sys.argv[1]]
except IndexError:
    #root, dir, videos = os.walk('videos')
    for root, dir, files in os.walk('videos'):
        videos = files


for video in videos:
    fileName = video.replace('.mp4', '')
    print (fileName)
    #check_scene_change_times(fileName)
    #sys.exit()
    fps = determine_frame_rate(fileName)
    black_frame_DF = parse_black_frame_text(fileName, fps)
    scene_change_DF = parse_scene_change_text(fileName, 0.1)
    #print (black_frame_DF)
    #print ()
    
