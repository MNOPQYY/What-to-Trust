import numpy as np
import os
import io
import cv2
import json
from tqdm import tqdm

import torch
import sys
import csv

sys.path.append('/media/sdb/qyy/object_state_change/VideoLLaMA2')
import argparse
from videollama2 import model_init, mm_infer_prob
from videollama2.utils import disable_torch_init

import ast

def reorganize_string_format(input_str):
    output=input_str.replace("[","['")
    output=output.replace("]","']")
    output=output.replace(", ","', '")
    
    return output

def organize_knowledge(state_dict,obj_act_kn_file,appearance_kn_file,obj):
    #将某个某提对应的三种知识，以状态为分类进行整理，方便后续每个片段依次回答各个状态的知识符合度
    obj_act_kn_obj = obj_act_kn_file[obj]
    appearance_kn_obj = appearance_kn_file[obj]
    state_list = state_dict[obj]
    obj_kn_dict = {}
    for state in state_list:
        if state in list(obj_act_kn_obj.keys()):
            co_objs,acts = obj_act_kn_obj[state]
            co_objs = reorganize_string_format(co_objs)
            acts = reorganize_string_format(acts)
            # print(co_objs)
            # print(co_objs[0])
            co_objs = ast.literal_eval(co_objs)
            acts = ast.literal_eval(acts)
            # print(co_objs)
            # print(acts)
        if state in list(appearance_kn_obj.keys()):
            appearances = appearance_kn_obj[state][0]
            appearances = reorganize_string_format(appearances)
            appearances = ast.literal_eval(appearances)
        obj_kn_dict[state]=[appearances,co_objs,acts]
    
    return obj_kn_dict
    
def calculate_all_states(model,tokenizer,obj,static_kn_obj,audio_video_tensor):
    video_votes = {}
    kn_type_list = ['appearance','co-occurrence_objs','actions']
    for state in list(static_kn_obj.keys()):
        state_votes={}
        for idx in range(len(static_kn_obj[state])):
            kn_votes = []
            for item in static_kn_obj[state][idx]:
                question_template = ['In this video, does the appearance of the '+obj +' look like '+item+'?','Does this video contain '+item+'?','Does this video contain '+item+' action?']
                
                question = question_template[idx]+" Directly output 'Yes' or 'No' without additional statements!"
                # print(question)
                input_length,output_ids,output = mm_infer_prob(
                    audio_video_tensor,
                    question,
                    model=model,
                    tokenizer=tokenizer,
                    modal="video",
                    do_sample=False,
                )
                output_lower = [t.lower() for t in output]
                if 'yes' in output_lower:
                    word_index = output_lower.index('yes')
                    probabilities = torch.softmax(output_ids.scores[word_index], dim=-1)
                    state_prob = probabilities[0,output_ids.sequences[0][word_index]]
                    kn_votes.append(state_prob.item())
                    # state_probs[state] = str(state_prob.item())
                elif 'no' in output_lower:
                    word_index = output_lower.index('no')
                    probabilities = torch.softmax(output_ids.scores[word_index], dim=-1)
                    state_prob = 1.-probabilities[0,output_ids.sequences[0][word_index]]
                    kn_votes.append(state_prob.item())
                    # state_probs[state] = str(state_prob.item())
                else:
                    state_prob = 0.5
                    kn_votes.append(state_prob.item())
                    # state_probs[state] = str(state_prob.item())
                    
                
                # if 'No' in output or 'no' in output:
                    # kn_votes.append(0)
                # elif 'Yes' in output or 'yes' in output:
                    # kn_votes.append(1) 
                # else:
                    # kn_votes.append(0) 
            state_votes[kn_type_list[idx]]=kn_votes
        video_votes[state]=state_votes
        
    return video_votes

if __name__ == "__main__":
    
    state_dict = json.load(open('/media/sdb/qyy/object_state_change/ObjectStatefromAction/MOST_dataset/MOST_state_categories_dict.json','r'))
    
    # model_path='/media/sdb/qyy/huggingface_ckpts/models--DAMO-NLP-SG--VideoLLaMA2.1-7B-AV/snapshots/d944d4298cb0188c3d5406eff42929cb67731361'
    model_path='/media/sdb/qyy/huggingface_ckpts/models--DAMO-NLP-SG--VideoLLaMA2.1-7B-16F/snapshots/3d68d446b2eebfebbdaa90f34c483bd9730eb51b'
    modal_type='v'
    
    obj_act_kn_file = json.load(open('/media/sdb/qyy/object_state_change/QWEN2.5/obj_action_state_knowledge_QWEN.json','r'))
    appearance_kn_file = json.load(open('/media/sdb/qyy/object_state_change/QWEN2.5/appearance_state_knowledge_QWEN.json','r'))
    
    model, processor, tokenizer = model_init(model_path)
    preprocess = processor['audio' if modal_type == "a" else "video"]
    
    vid_dir = '/media/sdb/qyy/object_state_change/ObjectStatefromAction/MOST_dataset/segmented_clip_twice'
    obj_list = os.listdir(vid_dir)
    
    save_dict = {}
    if os.path.exists('withprob_static_knowledge_voting_results_videollama.json'):
        save_dict = json.load(open('withprob_static_knowledge_voting_results_videollama.json','r'))
    
    
    for obj in tqdm(obj_list,total=len(obj_list)):
        print(obj)
        static_kn_obj = organize_knowledge(state_dict,obj_act_kn_file,appearance_kn_file,obj)
        if obj not in list(save_dict.keys()):
            save_dict[obj] = {}
        vid_list = os.listdir(os.path.join(vid_dir,obj))
        
        for vid in tqdm(vid_list,total=len(vid_list)):
            if vid not in list(save_dict[obj].keys()):
                save_dict[obj][vid] = {}
            csv_path = os.path.join(vid_dir,obj,vid,'scene_segments.csv')
            clip_dict = {}
            with open(csv_path, mode='r', newline='') as csvfile:
              csvreader = csv.DictReader(csvfile)
              for row in csvreader:
                scene_index = row['scene_index']
                scene_index = int(scene_index.split('Scene-')[-1])#+1
                start_time = row['start_time']
                end_time = row['end_time']
                clip_dict[scene_index] = [start_time,end_time]
            seg_list = list(clip_dict.keys())
            for scene_id in tqdm(range(len(seg_list)),total=len(seg_list)):
                
                clip = ('-').join([vid,"Scene",f"{scene_id+1:03}"])+'.mp4'
                if clip in list(save_dict[obj][vid].keys()):
                    continue
                vid_path = os.path.join(vid_dir,obj,vid,clip)
                # print(vid_path)
                
                if modal_type == "a":
                    audio_video_tensor = preprocess(vid_path)
                else:
                    audio_video_tensor = preprocess(vid_path, va=True if modal_type == "av" else False)
                
                static_voting_results = calculate_all_states(model,tokenizer,obj,static_kn_obj,audio_video_tensor) #将每个元素构成问题，询问VLM，得到是否存在的0，1结果，对应原始item是否出现。最终根据平均分作为静态投票结果   
                
                save_dict[obj][vid][clip]={'votes':static_voting_results,'segment':clip_dict[seg_list[scene_id]]}
                
                json.dump(save_dict,open('withprob_static_knowledge_voting_results_videollama.json','w'))
                    