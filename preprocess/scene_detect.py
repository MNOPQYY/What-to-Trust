from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg
import os
from tqdm import tqdm
import csv
import cv2 

def split_video_into_scenes(video_path,cur_dir, threshold=27.0):
    # Open our video, create a scene manager, and add a detector.
    video = open_video(video_path)
    vid_nm = video_path.split('/')[-1].split('.')[0]
    scene_manager = SceneManager()
    
    Detector = AdaptiveDetector(adaptive_threshold=5,
                                    min_scene_len=15,
                                    window_width=4,
                                    min_content_val=6)
    scene_manager.add_detector(Detector)
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list(start_in_scene=True)
    time_list = scene_manager.get_scene_list()
    
    segments = []
    for i in range(len(time_list) - 1):
        start_time = time_list[i][0].get_seconds()
        end_time = time_list[i + 1][0].get_seconds() 
        segments.append({'scene_index':'Scene-'+str(i),'start_time': start_time, 'end_time': end_time})
    if len(scene_list) > 0:
        start_time = scene_list[-1][0].get_seconds() 
        end_time = video.duration.get_seconds() 
        segments.append({'scene_index':'Scene-'+str(len(time_list)-1),'start_time': start_time, 'end_time': end_time})
    with open(cur_dir+'/'+vid_nm+'/'+'scene_segments.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['scene_index','start_time', 'end_time'])
        writer.writeheader()
        writer.writerows(segments)
    split_video_ffmpeg(video_path, scene_list, output_file_template=cur_dir+'/$VIDEO_NAME/$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4', show_progress=False)

if __name__ == "__main__":
    vid_dir='data/download_videos'
    vid_types = os.listdir(vid_dir)
    os.makedirs('data/segmented_clip',exist_ok=True)

    for vid_type in tqdm(vid_types,total=len(vid_types)):
        cur_dir = 'data/segmented_clip/'+vid_type
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        vids = os.listdir(vid_dir+'/'+vid_type)
        for vid in tqdm(vids,total=len(vids)):
            new_dir = cur_dir+'/'+vid.split('.')[0]
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            split_video_into_scenes(vid_dir+'/'+vid_type+'/'+vid,cur_dir)
            
