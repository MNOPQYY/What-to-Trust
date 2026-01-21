import sys
import argparse
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

import os
import json
from tqdm import tqdm
import csv

def inference_AU(args):

    model_path = args.model_path
    model, processor, tokenizer = model_init(model_path)

    if args.modal_type == "a":
        model.model.vision_tower = None
    elif args.modal_type == "v":
        model.model.audio_tower = None
    elif args.modal_type == "av":
        pass
    else:
        raise NotImplementedError
    preprocess = processor['audio' if args.modal_type == "a" else "video"]
    save_dict = {}
    dataset_dir = 'data/segmented_clip/'
    prompts = {'apple':'cooking apples','egg':'cooking eggs','flour':'cooking flour','wire':'operating wires','shirt':'operating shirts','tire':'operating tires'}
        
    for category in os.listdir(dataset_dir):
      folder_path = os.path.join(dataset_dir,category)
      if category not in save_dict.keys():
        save_dict[category]={}
      for vid_name in tqdm(os.listdir(folder_path),total=len(os.listdir(folder_path))):
          print(vid_name)
          vid_path = os.path.join(folder_path, vid_name)
          csv_path = os.path.join(vid_path,'scene_segments.csv')
          clip_dict = {}
          with open(csv_path, mode='r', newline='') as csvfile:
            csvreader = csv.DictReader(csvfile)
            
            for row in csvreader:
                scene_index = row['scene_index']
                scene_index = int(scene_index.split('Scene-')[-1])+1
                start_time = row['start_time']
                end_time = row['end_time']
                clip_dict[scene_index] = [start_time,end_time]
          if vid_name in save_dict[category].keys():
            continue
          save_dict[category][vid_name]={}
          
          previous_cap = 'Given the topic of '+ prompts[category]
          for scene_id in tqdm(range(len(os.listdir(vid_path))-1),total=len(os.listdir(vid_path))-1):
            scene_id+=1
            clip = ('-').join([vid_name,"Scene",f"{scene_id:03}"])+'.mp4'
            file_path = os.path.join(vid_path, clip)
            if args.modal_type == "a":
                audio_video_tensor = preprocess(file_path)
            else:
                audio_video_tensor = preprocess(file_path, va=True if args.modal_type == "av" else False)
            question = previous_cap+f", describe this video."
            output = mm_infer(
                audio_video_tensor,
                question,
                model=model,
                tokenizer=tokenizer,
                modal='audio' if args.modal_type == "a" else "video",
                do_sample=False,
            )
            save_dict[category][vid_name][clip]={'caption':output,'segment':clip_dict[scene_id]}
            
          json.dump(save_dict,open('data/MOST_clip_captions_with_obj_category.json','w'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=False, default='')
    parser.add_argument('--modal-type', choices=["a", "v", "av"], default='av', help='', required=False)
    args = parser.parse_args()

    inference_AU(args)

