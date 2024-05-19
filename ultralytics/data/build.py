# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed

from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset
from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
from ultralytics.data.utils import IMG_FORMATS, PIN_MEMORY, VID_FORMATS
from ultralytics.utils import LINUX, RANK, colorstr
from ultralytics.utils.checks import check_file


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        if not isinstance(kwargs['sampler'], PKSampler):
            super().__init__(*args, **kwargs)
            object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        else:
            kwargs["batch_sampler"] = _RepeatSampler(kwargs["sampler"])
            kwargs["sampler"] = None
            kwargs["batch_size"] = 1
            kwargs["drop_last"] = None
            super().__init__(*args, **kwargs)
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        """
        Reset iterator.

        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    """Build YOLO Dataset."""
    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_grounding(cfg, img_path, json_file, batch, mode="train", rect=False, stride=32):
    """Build YOLO Dataset."""
    return GroundingDataset(
        img_path=img_path,
        json_file=json_file,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


class PKSampler(torch.utils.data.Sampler):
    """First, randomly sample P identities.
        Then for each identity randomly sample K instances.
        Therefore batch size equals to P * K, and the sampler called PKSampler.

    Args:
        dataset (Dataset): Dataset which contains list of (img_path, pid, camid))
        batch_size (int): batch size
        sample_per_id (int): number of instance(s) within an class
        shuffle (bool, optional): _description_. Defaults to True.
        id_list(list): list of (start_id, end_id, start_id, end_id) for set of ids to duplicated.
        ratio(list): list of (ratio1, ratio2..) the duplication number for ids in id_list.
        drop_last (bool, optional): whether to discard the data at the end. Defaults to True.
        sample_method (str, optional): sample method when generating prob_list. Defaults to "sample_avg_prob".
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 sample_per_id,
                 shuffle=True,
                 drop_last=True,
                 id_list=None,
                 ratio=None,
                 sample_method="sample_avg_prob"):
        # super().__init__(
        #    dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
        assert batch_size % sample_per_id == 0, \
            f"PKSampler configs error, sample_per_id({sample_per_id}) must be a divisor of batch_size({batch_size})."
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.drop_last = drop_last
        assert hasattr(self.dataset,
                       "labels"), "Dataset must have labels attribute."
        self.sample_per_label = sample_per_id
        self.label_dict = defaultdict(list)
        self.sample_method = sample_method
        for idx, label in enumerate(self.dataset.labels):
            self.label_dict[label].append(idx)
        self.label_list = list(self.label_dict)
        assert len(self.label_list) * self.sample_per_label > self.batch_size, \
            "batch size should be smaller than "
        if self.sample_method == "id_avg_prob":
            self.prob_list = np.array([1 / len(self.label_list)] *
                                      len(self.label_list))
        elif self.sample_method == "sample_avg_prob":
            counter = []
            for label_i in self.label_list:
                counter.append(len(self.label_dict[label_i]))
            self.prob_list = np.array(counter) / sum(counter)
        else:
            print(
                "PKSampler only support id_avg_prob and sample_avg_prob sample method, "
                "but receive {}.".format(self.sample_method))

        if id_list and ratio:
            assert len(id_list) % 2 == 0 and len(id_list) == len(ratio) * 2
            for i in range(len(self.prob_list)):
                for j in range(len(ratio)):
                    if i >= id_list[j * 2] and i <= id_list[j * 2 + 1]:
                        self.prob_list[i] = self.prob_list[i] * ratio[j]
                        break
            self.prob_list = self.prob_list / sum(self.prob_list)

        diff = np.abs(sum(self.prob_list) - 1)
        if diff > 0.00000001:
            self.prob_list[-1] = 1 - sum(self.prob_list[:-1])
            if self.prob_list[-1] > 1 or self.prob_list[-1] < 0:
                print("PKSampler prob list error")
            else:
                print(
                    "PKSampler: sum of prob list not equal to 1, diff is {}, change the last prob".
                    format(diff))

    def __len__(self):
        num_samples = len(self.dataset)
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size

    def __iter__(self):
        label_per_batch = self.batch_size // self.sample_per_label
        for _ in range(len(self)):
            batch_index = []
            batch_label_list = np.random.choice(
                self.label_list,
                size=label_per_batch,
                replace=False,
                p=self.prob_list)
            for label_i in batch_label_list:
                label_i_indexes = self.label_dict[label_i]
                if self.sample_per_label <= len(label_i_indexes):
                    batch_index.extend(
                        np.random.choice(
                            label_i_indexes,
                            size=self.sample_per_label,
                            replace=False))
                else:
                    batch_index.extend(
                        np.random.choice(
                            label_i_indexes,
                            size=self.sample_per_label,
                            replace=True))
            if not self.drop_last or len(batch_index) == self.batch_size:
                yield batch_index


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1, sampler_config=None):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), workers])  # number of workers
    #  nw = 0
    if sampler_config is None:
        sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = PKSampler(dataset, sampler_config["batch_size"], sampler_config["sample_per_id"],
                            sampler_config["drop_last"], sampler_config["shuffle"],
                            sampler_config["sample_method"])
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def check_source(source):
    """Check source type and return corresponding flag values."""
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS | VID_FORMATS)
        is_url = source.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source.lower() == "screen"
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # convert all list elements to PIL or np arrays
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError("Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict")

    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(source=None, batch=1, vid_stride=1, buffer=False):
    """
    Loads an inference source for object detection and applies necessary transformations.

    Args:
        source (str, Path, Tensor, PIL.Image, np.ndarray): The input source for inference.
        batch (int, optional): Batch size for dataloaders. Default is 1.
        vid_stride (int, optional): The frame interval for video sources. Default is 1.
        buffer (bool, optional): Determined whether stream frames will be buffered. Default is False.

    Returns:
        dataset (Dataset): A dataset object for the specified input source.
    """
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)

    # Dataloader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer)
    elif screenshot:
        dataset = LoadScreenshots(source)
    elif from_img:
        dataset = LoadPilAndNumpy(source)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride)

    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)

    return dataset
