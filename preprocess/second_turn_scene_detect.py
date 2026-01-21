from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg
import os
from tqdm import tqdm
import csv
import cv2 
import json

def split_video_into_scenes(idx,timestmp,segments_csv_list,vid_nm,video_path,cur_dir, threshold=27.0):
    # Open our video, create a scene manager, and add a detector.
    video = open_video(video_path)
    video_duration = video.duration.get_seconds()
    
    for i in range(5):
        tmp_threshold = 1.5-0.2*i
        tmp_window_width = 10+2*i
        scene_manager = SceneManager()
        Detector = AdaptiveDetector(adaptive_threshold=tmp_threshold,
                                        min_scene_len=60,
                                        window_width=tmp_window_width,
                                        min_content_val=6)
        scene_manager.add_detector(Detector)
        scene_manager.detect_scenes(video, show_progress=False)
        time_list = scene_manager.get_scene_list()
        if len(time_list)>1:
            break
    
    for i in range(len(time_list)):
        start_time = time_list[i][0].get_seconds()
        end_time = time_list[i][1].get_seconds() 
        segments_csv_list.append({'scene_index':'Scene-'+str(i+idx),'start_time': start_time+timestmp, 'end_time': end_time+timestmp})
    
    if time_list == []:
        source_file = video_path
        new_seg = ('-').join([vid,"Scene",f"{idx:03}"])+'.mp4'
        destination = os.path.join(cur_dir,new_seg)
        os.system(f'cp {source_file} {destination}')
        segments_csv_list.append({'scene_index':'Scene-'+str(idx),'start_time': timestmp, 'end_time': timestmp+video_duration})
    else:
      for i,scene in enumerate(time_list):
        split_video_ffmpeg(video_path, [scene], output_file_template=cur_dir+'/'+vid_nm+'-Scene-'+f"{i+idx:03}"+'.mp4', show_progress=False)    
    
    return segments_csv_list,video.duration.get_seconds()+timestmp

if __name__ == "__main__":
    vid_dir = 'data/segmented_clip/'
    cap_file = json.load(open('data/MOST_clip_captions_with_obj_category.json','r'))
    key_words=['then', 'also', 'additionally']
    vid_types = os.listdir(vid_dir)

    for vid_type in tqdm(vid_types,total=len(vid_types)):
        cur_dir = 'data/segmented_clip_twice/'+vid_type
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        vids = os.listdir(vid_dir+'/'+vid_type)
        for vid in tqdm(vids,total=len(vids)):
            new_dir = cur_dir+'/'+vid
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            seg_dir = os.path.join(vid_dir,vid_type,vid)
            seg_list = os.listdir(seg_dir)
            idx=1
            segments_csv_list = []
            start_time=0
            print(vid)
            for seg_idx in range(1,len(seg_list)):
                seg = ('-').join([vid,"Scene",f"{seg_idx:03}"])+'.mp4'
                cap = cap_file[vid_type][vid][seg]['caption']
                duration = cap_file[vid_type][vid][seg]['segment']
                detected_words = any(word.lower() in cap.lower() for word in key_words)
                seg_dur = float(duration[1])-float(duration[0])
                if detected_words or seg_dur>40:
                    segments_csv_list,timestmp = split_video_into_scenes(idx,start_time,segments_csv_list,vid,os.path.join(seg_dir,seg),new_dir)
                    idx = len(os.listdir(new_dir))+1
                    start_time = timestmp
                else:
                    source_file = os.path.join(seg_dir,seg)
                    new_seg = ('-').join([vid,"Scene",f"{idx:03}"])+'.mp4'
                    destination = os.path.join(new_dir,new_seg)
                    os.system(f'cp {source_file} {destination}')
                    segments_csv_list.append({'scene_index':'Scene-'+str(idx),'start_time': duration[0], 'end_time': duration[1]})
                    idx+=1
                    start_time = float(duration[1])
            with open(new_dir+'/'+'scene_segments.csv', mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['scene_index','start_time', 'end_time'])
                writer.writeheader()
                writer.writerows(segments_csv_list)