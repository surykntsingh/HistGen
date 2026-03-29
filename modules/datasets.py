import os
import json
import torch

from torch.utils.data import Dataset
import h5py

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.max_fea_length = args.max_fea_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = []
        reports = self.read_json_file(self.ann_path)[split]

        for r in reports:
            img_name = r['id']
            image_path = os.path.join(self.image_dir, f'{img_name}.h5')

            if not os.path.isfile(image_path):
                continue

            # quickfix
            if img_name == 'TCGA-A2-A1G0-01Z-00-DX1.9ECB0B8A-EF4E-45A9-82AC-EF36375DEF65':
                continue

            anno = r['report']
            report_ids = tokenizer(anno)

            if len(report_ids) < self.max_seq_length:
                padding = [0] * (self.max_seq_length-len(report_ids))
                report_ids.extend(padding)

            self.examples.append(
                {
                    'id':img_name,
                    'image_path': image_path,
                    'report': anno, 'split': self.split,
                    'ids':report_ids,
                    'mask': [1]*len(report_ids)
                }
            )




        # for i in range(len(self.examples)):
        #     self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
        #     #* Below is the code to generate the mask for the report
        #     #* such a mask is used to indicate the positions of actual tokens versus padding positions in a sequence.
        #     self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)
    
class PathologySingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        # image_path = os.path.join(self.image_dir, image_id + '.pt')
        image_path = example['image_path']

        try:
            with h5py.File(image_path, "r") as h5_file:
                # coords_np = h5_file["coords"][:]
                embeddings_np = h5_file["features"][:]

                # coords = torch.tensor(coords_np).float()
                image = torch.tensor(embeddings_np)
        except Exception as e:
            print(f'Problem with {image_path},\n {e}')
        image = image[:self.max_fea_length]
        report_ids = example['ids']
        report_masks = example['mask']

        seq_length = len(report_ids)

        # report_ids = example['ids']
        # report_masks = example['mask']
        # seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample