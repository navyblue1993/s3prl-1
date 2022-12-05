import json
from pdb import set_trace

train_path = "downstream/emotion/meta_data/Session1/train_meta_data.json"
test_path = "downstream/emotion/meta_data/Session1/test_meta_data.json"

with open(test_path, 'r') as f:
# with open(train_path, 'r') as f:
    # train_dict = json.load(f)
    test_dict = json.load(f)
    set_trace()