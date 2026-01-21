import sys
import argparse
from videollama2 import model_init, mm_infer_prob
from videollama2.utils import disable_torch_init

import os
import json
from tqdm import tqdm
import csv
import torch

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
   
    dataset_dir = 'data/segmented_clip_twice/'
    topic_file = json.load(open('data/topic_MOST_clip_video_llama.json','r'))
    state_dict = json.load(open('data/MOST_state_categories_dict.json','r'))
        
    for category in os.listdir(dataset_dir):
      folder_path = os.path.join(dataset_dir,category)
      if category not in save_dict.keys():
        save_dict[category]={}
      for vid_name in tqdm(os.listdir(folder_path),total=len(os.listdir(folder_path))):
          vid_path = os.path.join(folder_path, vid_name)
          csv_path = os.path.join(vid_path,'scene_segments.csv')
          clip_dict = {}
          with open(csv_path, mode='r', newline='') as csvfile:
            csvreader = csv.DictReader(csvfile)
            
            for row in csvreader:
                scene_index = row['scene_index']
                scene_index = int(scene_index.split('Scene-')[-1])#+1
                start_time = row['start_time']
                end_time = row['end_time']
                clip_dict[scene_index] = [start_time,end_time]
          if vid_name in save_dict[category].keys():
            continue
          save_dict[category][vid_name]={}

          for scene_id in tqdm(range(1,len(os.listdir(vid_path))-1),total=len(os.listdir(vid_path))-1):
            clip = ('-').join([vid_name,"Scene",f"{scene_id:03}"])+'.mp4'
            scene_id+=1
            clip_prev = ('-').join([vid_name,"Scene",f"{scene_id:03}"])+'.mp4'
            file_path = os.path.join(vid_path, clip_prev)
            if args.modal_type == "a":
                audio_video_tensor = preprocess(file_path)
            else:
                audio_video_tensor = preprocess(file_path, va=True if args.modal_type == "av" else False)
            states = state_dict[category]
            pred_states = []
            state_probs = {}
            vid_topic = topic_file[category][vid_name+'.mp4']['topic']
            vid_des = topic_file[category][vid_name+'.mp4']['description']

            for state in states:
                if category == 'flour':
                    question = "You'll see a small segment of an entire video with a topic of '"+vid_topic+"', whose content is about'"+vid_des+"' Based on the given segment, is it plausible for the "+category+" to be "+state+" before this operation in this video? Directly output 'yes' or 'no' without other statements." 
                else:
                    question = "This is a small segment of an entire video with a topic of '"+vid_topic+"'. Based on the possible previous step of this video segment, is it plausible for the "+category+" to be "+state+" before this step? Directly output 'yes' or 'no' without other statements." 
                input_length,output_ids,output = mm_infer_prob(
                    audio_video_tensor,
                    question,
                    model=model,
                    tokenizer=tokenizer,
                    modal='audio' if args.modal_type == "a" else "video",
                    do_sample=False,
                )
                output_lower = [t.lower() for t in output]
                if 'yes' in output_lower:
                    word_index = output_lower.index('yes')
                    probabilities = torch.softmax(output_ids.scores[word_index], dim=-1)
                    state_prob = probabilities[0,output_ids.sequences[0][word_index]]
                    pred_states.append(state)
                    state_probs[state] = str(state_prob.item())
                elif 'no' in output_lower:
                    word_index = output_lower.index('no')
                    probabilities = torch.softmax(output_ids.scores[word_index], dim=-1)
                    state_prob = 1.-probabilities[0,output_ids.sequences[0][word_index]]
                    state_probs[state] = str(state_prob.item())
                else:
                    state_prob = 0.5
                    state_probs[state] = str(state_prob.item())
            save_dict[category][vid_name][clip]={'caption':pred_states,'state_prob':state_probs,'segment':clip_dict[scene_id-1]}
          json.dump(save_dict,open('data/backward_temporal_dependency_scores.json','w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=False, default='')
    parser.add_argument('--modal-type', choices=["a", "v", "av"], default='v', help='', required=False)
    args = parser.parse_args()

    inference_AU(args)
    