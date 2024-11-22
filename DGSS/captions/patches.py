import json
import ipdb

new_patch = {}

with open('/workspace/ssd0/byeongcheol/DGSS/captions/patches_captions.json', 'r') as f:
    patch = json.load(f)

for key, value in patch.items():
    new_patch[key.replace('_img', '')] = value

with open('/workspace/ssd0/byeongcheol/DGSS/captions/patches_new_captions.json', 'w') as f:
    json.dump(new_patch, f)

ipdb.set_trace()