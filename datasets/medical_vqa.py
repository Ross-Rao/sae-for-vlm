import os
import json
import glob
import csv
import re
from PIL import Image
from torch.utils.data import Dataset


class OmniMedVQADataset(Dataset):

    def __init__(self, root, split='open', transform=None, modality_filter=None):
        self.root = root
        self.transform = transform
        self.items = []

        qa_dir = os.path.join(root, 'QA_information', 'Open-access')
        for f in sorted(glob.glob(os.path.join(qa_dir, '*.json'))):
            with open(f) as fp:
                for item in json.load(fp):
                    if modality_filter and item.get('modality_type') not in modality_filter:
                        continue
                    img_path = os.path.join(root, item['image_path'])
                    if os.path.exists(img_path):
                        self.items.append({**item, '_img_path': img_path})

        for item in self.items:
            for letter, key in [('A', 'option_A'), ('B', 'option_B'),
                                 ('C', 'option_C'), ('D', 'option_D')]:
                if item.get(key) == item['gt_answer']:
                    item['_answer_label'] = letter
                    break
            else:
                item['_answer_label'] = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item['_img_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, item

    def get_prompt(self, item):
        return (
            "Question: {}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n"
            "Answer with the option letter (A, B, C, or D) only."
        ).format(item['question'], item['option_A'], item['option_B'],
                 item['option_C'], item['option_D'])


class PMCVQADataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.items = []

        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = os.path.join(img_dir, row['Figure_path'].strip())
                if os.path.exists(img_path):
                    self.items.append({**row, '_img_path': img_path,
                                       '_answer_label': row['Answer_label'].strip()})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item['_img_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, item

    def get_prompt(self, item):
        def strip_prefix(s):
            return re.sub(r'^\s*[ABCD]:\s*', '', s).strip()
        return (
            "Question: {}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n"
            "Answer with the option letter (A, B, C, or D) only."
        ).format(item['Question'].strip(),
                 strip_prefix(item['Choice A']), strip_prefix(item['Choice B']),
                 strip_prefix(item['Choice C']), strip_prefix(item['Choice D']))


class MedicalImageOnlyDataset(Dataset):

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, _ = self.base[idx]
        return image, 0
