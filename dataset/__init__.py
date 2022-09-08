from xml.etree.ElementTree import PI
from .kara.KaraDataset import make_kara_dataset
from .Pretrain.RawDataset import RawDataset
from .PileDataset import PileDataset
from .FFRecordDataset import FFRecordDataset

dataset_list = {
    "kara": make_kara_dataset,
    "Raw": RawDataset,
    "Pile": PileDataset,
    "FFRecord": FFRecordDataset
}