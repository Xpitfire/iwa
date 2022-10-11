import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random
import torch.utils.data as data
import torchvision.transforms.functional as F
import requests
from multiprocessing import Manager
from tqdm import tqdm
from PIL import Image
import tarfile
import zipfile
from torch.utils.data import Dataset, DataLoader
from hydra.utils import get_original_cwd


datasets = None
dataset_caches = None
img_compose = None
train_resize = None
test_resize = None
label_compose = lambda x: torch.from_numpy(x).squeeze()
def data_transform(mode):
    def trans(x, y):
        x = img_compose[mode](x)
        y = label_compose(y)
        return x, y
    return trans


class ListDataset(data.Dataset):
    '''Load image/labels from a list file.
    The list file is like:
      a.jpg label ...
    '''
    def __init__(self, root, list_file, domain, transform=None):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str/[str]) path to index file.
          transform: (function) image/box transforms.
        '''
        self.root = root
        self.transform = transform
        self.domain = domain

        self.fnames = []
        self.labels = []
        self.mean = None
        self.std = None
        self.normalize = None

        if isinstance(list_file, list):
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            self.labels.append(np.array([int(splited[1])]))

    def set_normalization(self, mean, std):
        """Set the dataset normalization parameters to get zero mean
        and unit variance.
        """
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(self.mean, self.std)

    def __getitem__(self, idx):
        '''Load image.
        Args:
          idx: (int) image index.
        Returns:
          img: (tensor) image tensor.
          boxes: (tensor) bounding box targets.
          labels: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img = None
        with Image.open(os.path.join(self.root, fname)) as i:
            if i.mode != 'RGB':
                i = i.convert('RGB')
            img = i
        labels = self.labels[idx]
        if self.transform:
            img, labels = self.transform(img, labels)
            if self.normalize:
                img = self.normalize(img)
        return img, labels

    def __len__(self):
        return self.num_imgs


def download_and_extract_tar_data(tmpfile, url, data_path, exist_ok=True):
    if not exist_ok and os.path.exists(tmpfile):
        raise AssertionError('File already exists!')
    if not os.path.exists(tmpfile):
        print('Downloading data... please wait')
        r = requests.get(url, allow_redirects=False)
        with open(tmpfile, 'wb') as tmp:
            tmp.write(r.content)
        print('Downloading data... completed')
    print('Extracting tar file... please wait')
    with tarfile.open(tmpfile) as test_data_tar:
        test_data_tar.extractall(data_path)
    print('Extracting tar file... completed')


def download_and_extract_zip_data(tmpfile, url, data_path, exist_ok=True):
    if not exist_ok and os.path.exists(tmpfile):
        raise AssertionError('File already exists!')
    if not os.path.exists(tmpfile):
        print('Downloading data... please wait')
        r = requests.get(url, allow_redirects=False)
        with open(tmpfile, 'wb') as tmp:
            tmp.write(r.content)
        print('Downloading data... completed')
    print('Extracting zip file... please wait')
    with zipfile.ZipFile(tmpfile, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    print('Extracting zip file... completed')


def download_file(file, url, exist_ok=True):
    if not exist_ok and os.path.exists(file):
        raise AssertionError('File already exists!')
    if not os.path.exists(file):
        print('Downloading file... please wait')
        r = requests.get(url, allow_redirects=False)
        with open(file, 'wb') as tmp:
            tmp.write(r.content)
        print('Downloading file... completed')


def download_domain_task(domain, config):
    os.makedirs(os.path.join(get_original_cwd(), config.dataloader.DomainNet.data_root), exist_ok=True)
    domain_root_dir = os.path.join(get_original_cwd(),  config.dataloader.DomainNet.data_root, domain)
    os.makedirs(domain_root_dir, exist_ok=True)
    print(f'Downloading data for domain: {domain}')

    test_data_path = os.path.join(domain_root_dir, config.dataloader.DomainNet.test_dir)
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path, exist_ok=True)
        tmpfile = os.path.join(test_data_path, 'test.txt')
        download_file(tmpfile, config.dataloader.DomainNet[f'{domain}_test_url'])

    train_data_path = os.path.join(domain_root_dir, config.dataloader.DomainNet.train_dir)
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path, exist_ok=True)
        tmpfile = os.path.join(train_data_path, 'train.txt')
        download_file(tmpfile, config.dataloader.DomainNet[f'{domain}_train_url'])
    tmpfile = os.path.join(train_data_path, 'train.txt')
    # BUG fix for DomainNet dataset missign class in `painting` domain
    if 'painting' in train_data_path:
        with open(tmpfile, 'r') as f:
            lines = f.readlines()
        # check if the number of unique classes is already 345 
        if len(np.unique([int(line.split(' ')[1]) for line in lines])) < 345:
            new_lines = []
            fixed = False
            for line in lines:
                id = line.split(' ')[1]
                # if the number increased and it was not yet fixed then fix the missing values
                if int(id) > 327 and not fixed:
                    for add in fix_painting_missing_class.split('\n'):
                        new_lines.append(add + '\n')
                    fixed = True
                new_lines.append(line)
            with open(tmpfile, 'w') as f:
                f.writelines(new_lines)

    data_path = os.path.join(domain_root_dir, config.dataloader.DomainNet.data_dir)
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
        tmpfile = os.path.join(data_path, 'tmp.tar')
        download_and_extract_zip_data(tmpfile, config.dataloader.DomainNet[f'{domain}_data_url'], data_path)


class DatasetCache(object):
    def __init__(self, config, manager, domain, info=''):
        self.config = config
        self.domain = domain
        self.info = info
        self.manager = manager
        self._dict = manager.dict()
        print(f"Created Cache for domain: {domain},{info}")

    def is_cached(self, key):
        if not self.config.dataloader.full_data_in_memory:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        if not self.config.dataloader.full_data_in_memory:
            raise AttributeError('Data caching is disabled and get funciton is unavailable! Check your config.')
        return self._dict[str(key)]

    def cache(self, key, img, lbl):
        # only store if full data in memory is enabled
        if not self.config.dataloader.full_data_in_memory:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = (img, lbl)


class CustomDataset(Dataset):
    r"""Domain adaptation version of the moon dataset object to iterate and collect samples.
    """
    def __init__(self, data, cache, resize=None, transform=None):
        self.resize = resize
        self.transform = transform
        self.data = data
        self.data_cache = cache
        self.len = len(self.data.labels)

    @property
    def domain(self):
        return self.data.domain

    def reset_memory(self):
        self.data_cache.reset()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        idx = idx % len(self.data.labels)

        if self.data_cache.is_cached(idx):
            x, y = self.data_cache.get(idx)
        else:
            fname = self.data.fnames[idx]
            x = None
            with Image.open(os.path.join(self.data.root, fname)) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                x = img
                if self.resize:
                    x = self.resize(x)
            y = self.data.labels[idx]
            self.data_cache.cache(idx, x, y)

        if self.transform:
            x, y = self.transform(x, y)
            if self.data.normalize:
                x = self.data.normalize(x)
        
        return x, y


normalization_constants = {
    'clipart': ([0.7394421696662903, 0.7194972038269043, 0.6864858269691467], 
                [0.10284537076950073, 0.09482032060623169, 0.10145732015371323]),
    'infograph': ([0.6881173253059387, 0.696143627166748, 0.6643316149711609], 
                [0.07690685242414474, 0.06838201731443405, 0.07345311343669891]),
    'painting': ([0.5736592411994934, 0.5455572009086609, 0.5067537426948547], 
                [0.06766611337661743, 0.06310966610908508, 0.07177083194255829]),
    'quickdraw': ([0.9524446725845337, 0.9524446725845337, 0.9524446725845337], 
                [0.02918548509478569, 0.02918548509478569, 0.02918548509478569]),
    'real': ([0.6065871715545654, 0.5897215008735657, 0.5563891530036926], 
            [0.0787251889705658, 0.07439002394676208, 0.08104723691940308]),
    'sketch': ([0.8325340747833252, 0.826887309551239, 0.8179639577865601], 
            [0.0806773379445076,   0.08035894483327866,   0.08151522278785706]),
    'imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
}


def compute_dataset_stats(domain, dataset):
    print('Domain:', domain)
    means = {'r': [], 'g': [], 'b': []}
    stds = {'r': [], 'g': [], 'b': []}
    for img, lbl in DataLoader(dataset, batch_size=1):
        img = img.squeeze()
        means['r'].append(img[0].mean())
        means['g'].append(img[1].mean())
        means['b'].append(img[2].mean())
        stds['r'].append(img[0].std())
        stds['g'].append(img[1].std())
        stds['b'].append(img[2].std())
    print(f"[{torch.mean(torch.stack(means['r']))}, {torch.mean(torch.stack(means['g']))}, {torch.mean(torch.stack(means['b']))}], \
            [{torch.std(torch.stack(stds['r']))}, {torch.std(torch.stack(stds['g']))}, {torch.std(torch.stack(stds['b']))}]")


def invert_imagenet_normalization(img_tensor):
    mean = torch.tensor(normalization_constants['imagenet'][0]).view(-1, 1, 1).expand_as(img_tensor)
    std = torch.tensor(normalization_constants['imagenet'][1]).view(-1, 1, 1).expand_as(img_tensor)
    return (img_tensor*std)+mean


class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')


fix_painting_missing_class = """painting/t-shirt/painting_328_000002.jpg 327
painting/t-shirt/painting_328_000004.jpg 327
painting/t-shirt/painting_328_000005.jpg 327
painting/t-shirt/painting_328_000006.jpg 327
painting/t-shirt/painting_328_000007.jpg 327
painting/t-shirt/painting_328_000008.jpg 327
painting/t-shirt/painting_328_000009.jpg 327
painting/t-shirt/painting_328_000010.jpg 327
painting/t-shirt/painting_328_000011.jpg 327
painting/t-shirt/painting_328_000012.jpg 327"""


def create_combined_source_domain_adaptation_data(config):
    global datasets, dataset_caches, img_compose, train_resize, test_resize

    img_compose = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop((config.dataloader.DomainNet.crop, 
                                          config.dataloader.DomainNet.crop), 
                                          scale=config.dataloader.DomainNet.scale),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(config.dataloader.DomainNet.color_jitter_factor, 
                                   config.dataloader.DomainNet.color_jitter_factor,
                                   config.dataloader.DomainNet.color_jitter_factor),
            transforms.RandomRotation(config.dataloader.DomainNet.rotation_degrees),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.ToTensor()
        ]),
    }
    train_resize = transforms.Compose([
        SquarePad(),
        transforms.Resize(config.dataloader.DomainNet.image_size)
    ])
    test_resize = transforms.Compose([
        SquarePad(),
        transforms.Resize((config.dataloader.DomainNet.crop, 
                           config.dataloader.DomainNet.crop))
    ])

    os.makedirs(os.path.join(get_original_cwd(), config.dataloader.DomainNet.data_root), exist_ok=True)
    domains = config.dataloader.DomainNet.domains

    # donwloading data
    for domain in tqdm(domains):
        download_domain_task(domain, config)

    datasets = {}
    dataset_caches = {}
    manager = Manager()
    selected_classes = config.dataloader.DomainNet.selected_classes
    remap = {c:k for k, c in enumerate(selected_classes)}
    for i, domain in tqdm(enumerate(domains)):

        # compute source datasets
        if config.dataloader.combined_source:
            source_domains = list(filter(lambda x: x != domains[i], domains))
        else:
            source_domains = [domain]

        domain_root_dirs = []
        for s_domain in source_domains:
            d_root_dir = os.path.join(get_original_cwd(), config.dataloader.DomainNet.data_root, s_domain)
            domain_root_dirs.append(d_root_dir)

        # write a combined file for all sources
        source_domain_name = '-'.join(source_domains)
        train_combine_source_file = os.path.join(get_original_cwd(), config.dataloader.DomainNet.data_root, f"{source_domain_name}_train.txt")
        if not os.path.exists(train_combine_source_file):
            with open(train_combine_source_file, 'w') as outfile:
                for j, fname in enumerate(domain_root_dirs):
                    train_file = os.path.join(fname, config.dataloader.DomainNet.train_dir, 'train.txt')
                    with open(train_file) as infile:
                        for line in infile:
                            new_line = f"{source_domains[j]}/{config.dataloader.DomainNet.data_dir}/{line}"
                            outfile.write(new_line)
        test_combine_source_file = os.path.join(get_original_cwd(), config.dataloader.DomainNet.data_root, f"{source_domain_name}_test.txt")
        if not os.path.exists(test_combine_source_file):
            with open(test_combine_source_file, 'w') as outfile:
                for j, fname in enumerate(domain_root_dirs):
                    test_file = os.path.join(fname, config.dataloader.DomainNet.test_dir, 'test.txt')
                    with open(test_file) as infile:
                        for line in infile:
                            new_line = f"{source_domains[j]}/{config.dataloader.DomainNet.data_dir}/{line}"
                            outfile.write(new_line)
        train_dataset = ListDataset(root=os.path.join(get_original_cwd(), config.dataloader.DomainNet.data_root), 
                                    list_file=train_combine_source_file, 
                                    domain=source_domain_name)
        test_dataset = ListDataset(root=os.path.join(get_original_cwd(), config.dataloader.DomainNet.data_root), 
                                   list_file=test_combine_source_file, 
                                   domain=source_domain_name)

        means = []
        stds = []
        for s_domain in source_domains:
            stats_key = config.dataloader.pretraining_dataset if config.dataloader.pretrained else s_domain
            mean, std = normalization_constants[stats_key]
            means.append(mean)
            stds.append(std)
        mean = np.mean(means, axis=0)
        std = np.mean(stds, axis=0)
        train_dataset.set_normalization(mean, std)
        test_dataset.set_normalization(mean, std)

        num_classes = config.dataloader.DomainNet.num_classes
        if num_classes < 345:
            assert num_classes == len(selected_classes)
            cond = np.asarray([v in selected_classes for v in np.asarray(train_dataset.labels)]).nonzero()[0]
            train_dataset.fnames = [f for k, f in enumerate(train_dataset.fnames) if k in cond]
            train_dataset.labels = [np.array(remap[l.item()], dtype=l.dtype) for k, l in enumerate(train_dataset.labels) if k in cond]

            cond = np.asarray([v in selected_classes for v in np.asarray(test_dataset.labels)]).nonzero()[0]
            test_dataset.fnames = [f for k, f in enumerate(test_dataset.fnames) if k in cond]
            test_dataset.labels = [np.array(remap[l.item()], dtype=l.dtype) for k, l in enumerate(test_dataset.labels) if k in cond]
        
        datasets[source_domain_name] = (train_dataset, test_dataset)
        dataset_caches[source_domain_name] = (DatasetCache(config, manager, source_domain_name, 'train'), DatasetCache(config, manager, source_domain_name, 'test'))
    

def mdn_data_generator(data_path, domain_id, config, hparams):
    if dataset_caches is None:
        create_combined_source_domain_adaptation_data(config)
    batch_size = hparams["batch_size"]
    train_ds, test_ds = datasets[domain_id]
    train_cache, test_cache = dataset_caches[domain_id]
    train_loader = DataLoader(
        CustomDataset(train_ds,
                      train_cache,
                      resize=train_resize,
                      transform=data_transform('train')),
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.dataloader.DomainNet.num_workers
    )
    test_loader = DataLoader(
        CustomDataset(test_ds,
                      test_cache,
                      resize=test_resize,
                      transform=data_transform('test')),
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.dataloader.DomainNet.num_workers
    )
    return train_loader, test_loader
