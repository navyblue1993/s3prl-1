from pdb import set_trace
import pickle
import os
import json

out_emb_dir = "/ocean/projects/tra220029p/zjia1/IEMOCAP_full_release/Dataset/IEMOCAP/wavlm_finetune_embeddings"
if not os.path.isdir(out_emb_dir):
    os.mkdir(out_emb_dir)

# train
emb_train_path = "result/downstream/exp1/embedding_5_train.pkl"
with open(emb_train_path, "rb") as f:
    emb_train = pickle.load(f)

# data info
meta_path = './downstream/emotion/meta_data/Session1/train_meta_data.json'
with open(meta_path, 'r') as f:
    data = json.load(f)

meta_data = data['meta_data']
for info, emb in zip(meta_data, emb_train):
    audio_path = info['path']
    audio_name = audio_path.split('/')[-1][:-4]
    
    emb_name = audio_name + ".pkl"
    emb_path = os.path.join(out_emb_dir, emb_name)
    with open(emb_path, "wb") as f:
        pickle.dump(emb, f)

# test
emb_test_path = "result/downstream/exp1/embedding_5_test.pkl"
with open(emb_test_path, "rb") as f:
    emb_test = pickle.load(f)

# data info
meta_path = './downstream/emotion/meta_data/Session1/test_meta_data.json'
with open(meta_path, 'r') as f:
    data = json.load(f)

meta_data = data['meta_data']
for info, emb in zip(meta_data, emb_test):
    audio_path = info['path']
    audio_name = audio_path.split('/')[-1][:-4]
    
    emb_name = audio_name + ".pkl"
    emb_path = os.path.join(out_emb_dir, emb_name)
    with open(emb_path, "wb") as f:
        pickle.dump(emb, f)

print(len(emb_test))