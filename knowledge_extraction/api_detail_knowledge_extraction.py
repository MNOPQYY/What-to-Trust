import os
from openai import OpenAI
import json
from tqdm import tqdm

def chat_qwen(client, obj, states, state):
    other_states = states.copy()
    other_states.remove(state)
    prompt_a="Given candidate states of "+ obj+", including ['"+"', '".join(states)+"'], what objects are particularly commonly for indicating the existence of " +state+" "+ obj+"? GENERATE A LIST WITHOUT ADDITIONAL STATEMENTS, expected form is like[object1,object2,...]."
    prompt_b="Given candidate states of "+ obj+", including ['"+"', '".join(states)+"'], what operations are mostly likely to cause " +state+" "+ obj+"? GENERATE A LIST WITHOUT ADDITIONAL STATEMENTS, expected form is like[operation1,operation2,...]."
    prompt_c="Given candidate states of "+ obj+", including ['"+"', '".join(states)+"'], what are discriminative appearances of " +state+" "+ obj+"? GENERATE A LIST WITHOUT ADDITIONAL STATEMENTS, expected form is like[appearance 1, appearance 2, ...]."
    
    responses=[]
    
    for prompt in [prompt_a,prompt_b,prompt_c]:
        
        messages_user = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
            ]
   
        messages_user.append({"role": "user", "content": prompt})
        
        completion = client.chat.completions.create(
        model="qwen2.5-72b-instruct", 
        messages=messages_user
        )
        response = completion.choices[0].message.content
        responses.append(response)
    
    return responses


if __name__ == "__main__":    
    client = OpenAI(
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    state_dict = json.load(open('data/MOST_state_categories_dict.json','r'))
    
    objs = list(state_dict.keys())
    
    save_dict = {}
    
    for obj in tqdm(objs,total=len(objs)):
        states = state_dict[obj]
        
        save_dict[obj]={}
        for state in tqdm(states,total=len(states)):
            state_knowledge = chat_qwen(client,obj,states,state)
            save_dict[obj][state]=state_knowledge
            json.dump(save_dict,open('data/state_specific_knowledge_QWEN.json','w'))