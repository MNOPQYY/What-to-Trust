import sys

sys.path.insert(0,'knowledge_voting/CLIP/')
import clip
import torch
from PIL import Image
import json
from tqdm import tqdm
import numpy as np
import os


def get_clip_features(image_path):
    if type(image_path)==list:
        image_inputs = []
        for img in image_path:
            image = preprocess(Image.open(img)).cuda()
            image_inputs.append(image)      
    else:
        image_inputs=[preprocess(Image.open(image_path)).cuda()]
            
    with torch.no_grad():
        features = model.encode_image(torch.stack(image_inputs))
        
    return features / features.norm(dim=-1, keepdim=True)

if __name__ == "__main__":
    frame_path = 'data/frames_segmented_clip_twice/'
    kn_path = 'data/image_knowledge/'
    model, preprocess = clip.load("knowledge_voting/CLIP/clip_ckpt/ViT-B-32.pt", device="cuda")
    state_file = json.load(open('data/MOST_state_categories_dict.json','r'))
    out_dict = {}
    for obj in tqdm(state_file,total=len(state_file)):
        out_dict[obj]={}
        state_list = state_file[obj]
        test_vids = os.listdir(frame_path+obj)
        
        for vid in tqdm(test_vids,total=len(test_vids)):
            out_sim = {}
            out_dict[obj][vid]={}
            test_frame_list = [os.path.join(frame_path,obj,vid,i) for i in os.listdir(os.path.join(frame_path,obj,vid))]
            test_frame_features = get_clip_features(test_frame_list)
            for state in state_list:
                kn_img_path = os.path.join(kn_path,obj,state)
                kn_imgs = os.listdir(os.path.join(kn_path,obj,state))
                sorted_files = sorted(kn_imgs,key=lambda x: int(x.split('.')[0]))
                kn_img_list = [os.path.join(kn_img_path,img) for img in sorted_files[:10]]
                kn_img_features = get_clip_features(kn_img_list)
                similarity = test_frame_features.cpu().numpy() @ kn_img_features.cpu().numpy().T
                
                for idx,i in enumerate(os.listdir(os.path.join(frame_path,obj,vid))):
                    if i not in out_dict[obj][vid]:
                        out_dict[obj][vid][i]={'votes':{},'segment':[]}
                    out_dict[obj][vid][i]['votes'][state]=similarity[idx].tolist()
        json.dump(out_dict,open('data/kn_similarity_perscene.json','w'))