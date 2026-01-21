import json
import numpy as np
import ast
import os
from tqdm import tqdm
import csv
from collections import defaultdict
from functools import partial
import re
import random
import torch
import torch.nn.functional as F

def vid_similarity_cal(tensor_a,tensor_b):
    sim = np.linalg.norm(np.array(tensor_a)-np.array(tensor_b))
    return 1/(sim+1e-12)
    
def reorganize_string_format(input_str):
    output=input_str.replace("[","['")
    output=output.replace("]","']")
    output=output.replace(", ","', '")
    return output

def organize_knowledge(state_dict,obj_act_kn_file,img_kn_file,obj):
    obj_act_kn_obj = obj_act_kn_file[obj]
    img_kn_obj = img_kn_file[obj]
    state_list = state_dict[obj]
    obj_kn_dict = {}
    init_weight = {}
    kn_share = []
    for state in state_list:
        if state in list(obj_act_kn_obj.keys()):
            co_objs,acts, appearances = obj_act_kn_obj[state]
            co_objs = reorganize_string_format(co_objs)
            acts = reorganize_string_format(acts)
            appearances = reorganize_string_format(appearances)
            co_objs = ast.literal_eval(co_objs)
            acts = ast.literal_eval(acts)
            appearances = ast.literal_eval(appearances)
        if state in list(img_kn_obj.keys()):
            img_kn = img_kn_obj[state]
            
        obj_kn_dict[state]=img_kn+appearances+co_objs+acts
    return obj_kn_dict

class ZeroShotStateEM:
    def __init__(self, state_knowledge, backward_temporal_file, forward_temporal_file, state_index, ratio, epsilon=0.0001):

        self.states = list(state_index.keys())
        self.state_idx = state_index
        self.K = len(self.states)
        self.M = {s: len(feats) for s, feats in state_knowledge.items()}
        self.theta = {
            s: np.array([1.0]*self.M[s], dtype=np.float32) 
            for s in self.states
        }
        self.temporal_backward = backward_temporal_file
        self.temporal_forward = forward_temporal_file
        self.epsilon = epsilon
        self.ratio = ratio
            
    def estimate_step(self, video_obs):
        T = len(video_obs)
        K = self.K
        gamma = np.zeros((T, K))
        all_state_obs = []
        weighted_obs_segs = np.zeros((T,K))
        weighted_obs_segs_add_temp = np.zeros((T,K))
        for t in range(0,T):
            weighted_obs = np.zeros(len(list(self.state_idx.keys()))) 
            each_scene_obs = []
            for k in range(K):
                state = self.states[k]
                vid_obs_ori = video_obs[t][state]
                theta_k = np.asarray(self.theta[state], dtype=np.float32)
                tmp_obs = (vid_obs_ori*theta_k).tolist()
                vis_score = sum(tmp_obs)/len(tmp_obs) 
                each_scene_obs+=tmp_obs 
                weighted_obs[k]=vis_score 
            weighted_obs /= np.sum(weighted_obs) 
            weighted_obs_segs[t] = weighted_obs
            all_state_obs.append(each_scene_obs)
        
        for t in range(0, T):
            clip = ('-').join([vid,"Scene",f"{t+1:03}"])+'.mp4'
            weighted_obs = weighted_obs_segs[t]
            if t>0 and t<T-1:
                sim_prev = vid_similarity_cal(all_state_obs[t-1],all_state_obs[t])
                sim_next = vid_similarity_cal(all_state_obs[t],all_state_obs[t+1])
                back_temp_kn = self.temporal_backward[clip]['state_prob']
                backward_temporal_prob = np.zeros(len(list(self.state_idx.keys())))
                for s,prob in back_temp_kn.items():
                        backward_temporal_prob[self.state_idx[s]] = prob
                back_ratio = sim_next/(sim_prev+sim_next)
                back_score = backward_temporal_prob**back_ratio
                
                forward_temp_kn = self.temporal_forward[clip]['state_prob']
                forward_temporal_prob = np.zeros(len(list(self.state_idx.keys())))
                for s,prob in forward_temp_kn.items():
                        forward_temporal_prob[self.state_idx[s]] = prob
                forward_ratio = sim_prev/(sim_prev+sim_next)
                forward_score = forward_temporal_prob**forward_ratio
                        
            elif t==0:
                sim_next = vid_similarity_cal(all_state_obs[t],all_state_obs[t+1])
                back_temp_kn = self.temporal_backward[clip]['state_prob']
                backward_temporal_prob = np.zeros(len(list(self.state_idx.keys())))
                for s,prob in back_temp_kn.items():
                        backward_temporal_prob[self.state_idx[s]] = prob
                back_ratio = 1
                back_score = backward_temporal_prob**back_ratio
                forward_score = np.ones(len(list(self.state_idx.keys())))

            elif t==T-1:
                sim_prev = vid_similarity_cal(all_state_obs[t-1],all_state_obs[t])
                forward_temp_kn = self.temporal_forward[clip]['state_prob']
                forward_temporal_prob = np.zeros(len(list(self.state_idx.keys())))
                for s,prob in forward_temp_kn.items():
                        forward_temporal_prob[self.state_idx[s]] = prob
                forward_ratio = 1
                forward_score = forward_temporal_prob**forward_ratio
                back_score = np.ones(len(list(self.state_idx.keys())))
            
            temporal_elem = forward_score*weighted_obs*back_score/sum(forward_score*weighted_obs*back_score)
            weighted_obs = self.ratio*(temporal_elem)+(1-self.ratio)*weighted_obs
            weighted_obs /= np.sum(weighted_obs) 
            weighted_obs_segs_add_temp[t] = weighted_obs
        return weighted_obs_segs_add_temp
        

    def update_step(self, video_gamma, video_obs,iter, alpha=1, beta=2):
        theta_sums = {s: defaultdict(float) for s in self.states}
        for t in range(len(video_gamma)): 
            for k, gamma_tk in enumerate(video_gamma[t]):
                state = self.states[k]
                obs = video_obs[t][state]
                for m in range(self.M[state]):
                    theta_sums[state][m] += gamma_tk * obs[m]
                theta_sums[state]['total'] += gamma_tk

        global_activation = defaultdict(float)
        total_frames = len(video_obs)
        for obs in video_obs:
            for state, features in obs.items():
                for m, val in enumerate(features):
                    global_activation[(state,m)] += val
                    
        global_activation = {k:v/total_frames for k,v in global_activation.items()}
        for state in self.states:
            M_state = self.M[state] 
            new_theta = np.zeros(M_state)
            for m in range(M_state):
                freq = global_activation.get((state,m), 0)
                penalty = np.exp(-2 * freq) 
                numer = (theta_sums[state][m] + alpha) * penalty
                denom = theta_sums[state]['total'] + alpha + beta
                new_theta[m] = numer / denom
            lr = 1 / (1 + iter)
            self.theta[state] = (1-lr)*self.theta[state] + (lr)*new_theta 

    def estimate_update_iteration(self, video_obs, max_iters=50, tol=1e-4):
        prev_theta = {s: np.copy(v) for s,v in self.theta.items()}
        history = {'theta': [], 'gamma': []}
        
        for iter in range(max_iters):
            gamma = self.estimate_step(video_obs)
            self.update_step(gamma, video_obs,iter)
            
            history['theta'].append({s: np.copy(v) for s,v in self.theta.items()})
            history['gamma'].append(gamma)
            
            delta = max(np.max(np.abs(self.theta[s] - prev_theta[s])) for s in self.states)
            
            if delta < tol:
                print(f"Converge at the {iter}-th iteration.")
                break
            prev_theta = {s: np.copy(v) for s,v in self.theta.items()}            
        return history


if __name__ == "__main__":
    
    temporal_ratio = 0.3
    state_dict = json.load(open('data/MOST_state_categories_dict.json','r'))
    textual_kn_file = json.load(open('data/state_specific_knowledge_QWEN.json','r'))
    visual_kn_file = {} 
    objs = list(state_dict.keys())
    img_kn_path = 'data/image_knowledge'
    for obj in objs:
        states_path = os.path.join(img_kn_path,obj)
        states = os.listdir(states_path)
        visual_kn_file[obj]={}
        for state in states:
            kn_imgs = os.listdir(os.path.join(states_path,state))
            sorted_files = sorted(kn_imgs,key=lambda x: int(x.split('.')[0]))
            kn_img_list = [os.path.join(states_path,state,img) for img in sorted_files[:10]]
            visual_kn_file[obj][state]=kn_img_list

    img_kn_voting_results = json.load(open('data/kn_similarity_perscene.json','r'))
    voting_results = json.load(open('data/withprob_static_knowledge_voting_results_videollama.json','r'))

    forward_temporal_file = json.load(open('data/forward_temporal_dependency_scores.json','r'))
    backward_temporal_file = json.load(open('data/backward_temporal_dependency_scores.json','r'))

    vid_dir = 'datas/segmented_clip_twice'
    obj_list = os.listdir(vid_dir)
    save_dict = {}
    previous_states=[]
    for obj in tqdm(obj_list,total=len(obj_list)):        
        vid_list = os.listdir(os.path.join(vid_dir,obj))
        state_knowledge = organize_knowledge(state_dict,visual_kn_file,textual_kn_file,obj)
        
        state_index = {state: i for i, state in enumerate(state_dict[obj])}
        
        save_dict[obj] = {}
        
        for vid in tqdm(vid_list,total=len(vid_list)):
            vid_obs_list=[]            
            previous_states = []
            save_dict[obj][vid] = {}
            csv_path = os.path.join(vid_dir,obj,vid,'scene_segments.csv')
            clip_dict = {}
            with open(csv_path, mode='r', newline='') as csvfile:
              csvreader = csv.DictReader(csvfile)
              for row in csvreader:
                scene_index = row['scene_index']
                scene_index = int(scene_index.split('Scene-')[-1])
                start_time = row['start_time']
                end_time = row['end_time']
                clip_dict[scene_index] = [start_time,end_time]
            seg_list = list(clip_dict.keys())
            prev_count_tensor = np.zeros(len(list(state_index.keys())))
            prev_add_tensor = np.zeros(len(list(state_index.keys())))
            
            em = ZeroShotStateEM(
                state_knowledge=state_knowledge,
                backward_temporal_file=backward_temporal_file[obj][vid],
                forward_temporal_file=forward_temporal_file[obj][vid],
                state_index=state_index,
                ratio=temporal_ratio,
                epsilon=0.00001
            )
        
            for scene_id in tqdm(range(len(seg_list)),total=len(seg_list)):
                vid_obs_dict = {}                
                clip = ('-').join([vid,"Scene",f"{scene_id+1:03}"])+'.mp4'
                prev_clip = ('-').join([vid,"Scene",f"{scene_id:03}"])+'.mp4'
                next_clip = ('-').join([vid,"Scene",f"{scene_id+2:03}"])+'.mp4'
                frame = ('-').join([vid,"Scene",f"{scene_id+1:03}"])+'.jpg'
                prev_frame = ('-').join([vid,"Scene",f"{scene_id:03}"])+'.jpg'
                next_frame = ('-').join([vid,"Scene",f"{scene_id+1:03}"])+'.jpg'
                                
                clip_votes = voting_results[obj][vid][clip]['votes']
                img_votes = img_kn_voting_results[obj][vid][frame]['votes']
                segments = voting_results[obj][vid][clip]['segment']
                cate_count_tensor=np.zeros(len(list(state_index.keys())))
                cate_add_tensor=np.zeros(len(list(state_index.keys())))
                cate_count_dict = {}
                tmp_obs_share=[]
                obs_prev_allstates=[]
                obs_next_allstates=[]
                obs_curr_allstates=[]
                for state in list(state_index.keys()):
                    state_votes = clip_votes[state]
                    img_state_votes = img_votes[state]
                    
                    kn_nm_list = ['appearance','co-occurrence_objs','actions']
                    tmp_obs = img_state_votes
                    tmp_obs_sim = img_state_votes
                    tmp_obs_sim_next = img_state_votes
                    for cate in kn_nm_list:
                        tmp_obs = tmp_obs+state_votes[cate]
                    vid_obs_dict[state]=tmp_obs.copy()
                vid_obs_list.append(vid_obs_dict)
                
            history = em.estimate_update_iteration(vid_obs_list, max_iters=20)
            cate_count_tensor_all = history['gamma'][-1]
            for scene_id in tqdm(range(len(seg_list)),total=len(seg_list)):
                cate_count_dict = {}
                clip = ('-').join([vid,"Scene",f"{scene_id+1:03}"])+'.mp4'
                frame = ('-').join([vid,"Scene",f"{scene_id+1:03}"])+'.jpg'
                                
                segments = voting_results[obj][vid][clip]['segment']
                
                cate_count_tensor = cate_count_tensor_all[scene_id]
                              
                for state in state_dict[obj]:
                    cate_count_dict[state] = cate_count_tensor[state_index[state]]
                
                save_dict[obj][vid][clip] = {'probs':cate_count_dict,'count':list(cate_count_tensor),'segment':segments}
                
            json.dump(save_dict,open('object_state_inference.json','w'))
            
