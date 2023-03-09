'''

@author: Dr. Kangcheng Liu

'''

import torch

from datasets.transforms import transforms

import numpy as np

collate_fn = transforms.cfl_collate_fn_factory(0)

def vox_moco_collator(batch):
    batch_size = len(batch)
    
    data_point = [x["data"] for x in batch]
    data_moco = [x["data_moco"] for x in batch]
    # labels are repeated N+1 times but they are the same
    labels = [int(x["label"][0]) for x in batch]
    labels = torch.LongTensor(labels).squeeze()

    # data valid is repeated N+1 times but they are the same
    data_valid = torch.BoolTensor([x["data_valid"][0] for x in batch])

    vox_moco = collate_fn([data_moco[i][0] for i in range(batch_size)])
    vox = collate_fn([data_point[i][0] for i in range(batch_size)])
    
    output_batch = {
        "vox": vox,
        "vox_moco": vox_moco,
        "label": labels,
        "data_valid": data_valid,
    }
    
    return output_batch
