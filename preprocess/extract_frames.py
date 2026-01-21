import ffmpeg
import os
from tqdm import tqdm
import sys
import subprocess

def extract_frames_uniform():
  vid_path = 'data/segmented_clip_twice/'
  objs = os.listdir(vid_path)
  vid_names = os.listdir(vid_path)
  path='data/frames_segmented_clip_twice/'
  os.makedirs(path,exist_ok=True)
  for obj in objs:
    obj_path = os.path.join(path,obj)
    if not os.path.exists(obj_path):
      os.mkdir(obj_path)
    vid_obj_path = os.path.join(vid_path,obj)
    for vid in os.listdir(vid_obj_path):
        vids = os.path.join(vid_obj_path,vid)
        scenes = os.listdir(vids)
        output_vid_path = os.path.join(obj_path,vid)
        if not os.path.exists(output_vid_path):
            os.mkdir(output_vid_path)
        for scene in scenes:
            if scene.split('.')[-1]=='csv':
                continue
            scene_nm = scene.split('.')[0]+'.jpg'
            probe = ffmpeg.probe(os.path.join(vids,scene))
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if video_stream:
                duration = video_stream['duration']
                timestamp = float(duration)/2
            else:
                timestamp = 0.1
            subprocess.run(['ffmpeg', '-i', os.path.join(vids,scene), '-ss', str(timestamp), '-vframes', '1', os.path.join(output_vid_path,scene_nm)])

if __name__ == "__main__":
    extract_frames_uniform()