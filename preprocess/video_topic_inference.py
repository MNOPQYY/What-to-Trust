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
    model, processor, tokenizer = model_init(model_path,num_frames=20)

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
    
    dataset_dir = 'data/download_videos/'
    
    state_dict = json.load(open('data/MOST_state_categories_dict.json','r'))
        
    for category in os.listdir(dataset_dir):
      folder_path = os.path.join(dataset_dir,category)
      if category not in save_dict.keys():
        save_dict[category]={}
      for vid_name in tqdm(os.listdir(folder_path),total=len(os.listdir(folder_path))):
          print(vid_name)
          vid_path = os.path.join(folder_path, vid_name)
          if vid_name in save_dict[category].keys():
            continue
          save_dict[category][vid_name]={}
          
          prompt_topic = 'Generate a title reflecting the main content of this '+category+'-related video.' 
          prompt_des = 'Generate a description of this '+category+'-related video.' 
          
          if args.modal_type == "a":
            audio_video_tensor = preprocess(vid_path)
          else:
            audio_video_tensor = preprocess(vid_path, va=True if args.modal_type == "av" else False)
          
          output_topic = mm_infer(
            audio_video_tensor,
            prompt_topic,
            model=model,
            tokenizer=tokenizer,
            modal='audio' if args.modal_type == "a" else "video",
            do_sample=False,
          )
          output_des = mm_infer(
            audio_video_tensor,
            prompt_des,
            model=model,
            tokenizer=tokenizer,
            modal='audio' if args.modal_type == "a" else "video",
            do_sample=False,
          )
          save_dict[category][vid_name]={'topic':output_topic, 'description':output_des}
          
          json.dump(save_dict,open('data/topic_MOST_clip_video_llama.json','w'))
          

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=False, default='')
    parser.add_argument('--modal-type', choices=["a", "v", "av"], default='av', help='', required=False)
    args = parser.parse_args()

    inference_AU(args)

