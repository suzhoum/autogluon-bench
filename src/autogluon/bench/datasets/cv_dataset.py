import logging
import os

import pandas as pd

from autogluon.bench.utils.dataset_utils import get_data_home_dir
from autogluon.common.loaders._utils import download, protected_zip_extraction

from .constants import _BINARY, _CATEGORICAL, _MULTICLASS
from .multimodal_dataset import BaseMultiModalDataset

s3_repo_url = "s3://automl-mm-bench/"

__all__ = [
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "Caltech256Dataset",
    "OxfordIIITPetDataset",
    "SUN397Dataset",
    "MNISTDataset",
    "BayerDataset",
    "BelgalogosDataset",
    "Cub200Dataset",
    "DescribableTexturesDataset",
    "EuropeanFloodDepthDataset",
    "FgvcAircraftsDataset",
    "Food101Dataset",
    "IfoodDataset",
    "KindleDataset",
    "MagneticTileDefectsDataset",
    "MalariaCellImagesDataset",
    "Mit67Dataset",
    "NikeDataset",
    "NWPUResisc45Dataset",
    "OpenSurfacesMinc2500Dataset",
    "OxfordFlowersDataset",
    "RedfinDataset",
    "SemartSchoolDataset",
    "StanfordCarsDataset",
    "StanfordDogsDataset",
]

logger = logging.getLogger(__name__)


class BaseVisionDataset(BaseMultiModalDataset):
    @property
    def feature_columns(self):
        return ["ImageID"]

    @property
    def label_columns(self):
        return ["LabelName"]

    @property
    def metric(self):
        return "acc"


class CIFAR10Dataset(BaseVisionDataset):
    _registry_name = "cifar10"
    _IMAGE_ZIP = _registry_name.upper() + "_ap.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/ap_datasets/" + _IMAGE_ZIP,
            "sha1sum": "cb5dc89eccbc185ae969f3eb18c17e67996edc71",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name.upper()) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )

        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._path,
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_anno.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            path_split = "validation" if self._split == "val" else self._split
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{path_split}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_MULTICLASS]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class CIFAR100Dataset(BaseVisionDataset):
    _registry_name = "cifar100"
    _IMAGE_ZIP = _registry_name.upper() + "_ap.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/ap_datasets/" + _IMAGE_ZIP,
            "sha1sum": "ff5288b094dd7c9dad6fec25102e80532d601200",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name.upper()) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )

        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._path,
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_anno.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            path_split = "validation" if self._split == "val" else self._split
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{path_split}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_MULTICLASS]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class Caltech256Dataset(BaseVisionDataset):
    _registry_name = "caltech256"
    _IMAGE_ZIP = _registry_name.capitalize() + "_ap.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/ap_datasets/" + _IMAGE_ZIP,
            "sha1sum": "3943fb2241152214bc4889a6a7446012ec254ed6",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name.capitalize()) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )

        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._path,
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_anno.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            path_split = "validation" if self._split == "val" else self._split
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{path_split}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_MULTICLASS]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class OxfordIIITPetDataset(BaseVisionDataset):
    _registry_name = "OxfordIIITPet"
    _IMAGE_ZIP = _registry_name + "_ap.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/ap_datasets/" + _IMAGE_ZIP,
            "sha1sum": "96be0eeeacbe97b3141af49383e7c441d816243e",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )

        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._path,
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_anno.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            path_split = "validation" if self._split == "val" else self._split
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{path_split}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_MULTICLASS]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class SUN397Dataset(BaseVisionDataset):
    _registry_name = "sun397"
    _IMAGE_ZIP = _registry_name.upper() + "_ap.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/ap_datasets/" + _IMAGE_ZIP,
            "sha1sum": "2e1fecdd68b656bb18d6b6977b44985a9bbce72e",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name.upper()) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )

        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._path,
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_anno.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = (
                    self._INFO["local_dir"] + self._registry_name.upper() + "/" + self._data[col].astype(str)
                )
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_MULTICLASS]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class MNISTDataset(BaseVisionDataset):
    _registry_name = "mnist"
    _IMAGE_ZIP = _registry_name + "_ap.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/ap_datasets/" + _IMAGE_ZIP,
            "sha1sum": "6d67a595809af12789286aa7e0c5f08dc3b4dfa7",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )

        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._path,
        )

        try:
            if self._split == "val":  # mismatch in dataset
                anno_split = "test"
            elif self._split == "test":
                anno_split = "val"
            else:
                anno_split = self._split
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{anno_split}_anno.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{self._split}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_MULTICLASS]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class BayerDataset(BaseVisionDataset):
    _registry_name = "bayer"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "ac88c2b5bead9bf9878430ddfabc9ed2a5fc1410",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "c4fae790a42f9fadfe8d5e070b3e30d398e3e83a",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._path,
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{self._split}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_BINARY]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _BINARY


class BelgalogosDataset(BaseVisionDataset):
    _registry_name = "belgalogos"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "74d25a2d10b5d00480b79de8a8ed2aacc24696ed",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "6f5b9b99a1e7bdfced58df728f80b8b266a87fe0",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{self._registry_name}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class Cub200Dataset(BaseVisionDataset):
    _registry_name = "cub200"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "00d6ac6314a189ad13353421be13f22d5819117b",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "2dca69d4015d5ad169f09ddccb2f28852d83881e",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{self._registry_name}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class DescribableTexturesDataset(BaseVisionDataset):
    _registry_name = "describabletextures"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "0b09f83ee2bba1240d5e7c92617b6bf195752507",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "27fe60a7a1d76c7af1262e27b96afd7f88059135",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{self._registry_name}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class EuropeanFloodDepthDataset(BaseVisionDataset):
    _registry_name = "europeanflooddepth"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "9a100c94023a23afc57ed217cd81808bd48817ea",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "8ddaf37bb8ef631a6b8a144e327d0685fa7b870f",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{self._registry_name}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_BINARY]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _BINARY


class FgvcAircraftsDataset(BaseVisionDataset):
    _registry_name = "fgvc_aircrafts"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "acf68f48dd1fd676362f5139a1b1d80e0559e38d",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "1d44eab29ce1fe668c6560a7588cb4f31ca24721",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{self._registry_name}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class Food101Dataset(BaseVisionDataset):
    _registry_name = "food101"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "f3f0d702763efd8a8633b446cf0cf3a6f4a9c6a7",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "d85ec18558731cf30745d0fd636cc11b9ef8cef7",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{self._registry_name}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class IfoodDataset(BaseVisionDataset):
    _registry_name = "ifood"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "35ba613456271528cf0370ddf5fdc39a5b830d4c",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "ff8681e54a2bb1e722b4aaf9a6846bd36159016a",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{self._registry_name}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class KindleDataset(BaseVisionDataset):
    _registry_name = "kindle"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "f50244247080c57f2f70776ccc75de5728113fb3",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "39fb796d0ed7348f93d7ee8d9e58bd5864b2bf59",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = (
                    self._INFO["local_dir"]
                    + f"{self._registry_name}/"
                    + f"{self._split}/"
                    + self._data[col].astype(str)
                )
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class MagneticTileDefectsDataset(BaseVisionDataset):
    _registry_name = "magnetictiledefects"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "4e3d48b07a5a443d3b3d0530f2a2089d3464c350",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "3a4d46e25389c4401e74adfe8e0a2c2432e7fc17",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{self._registry_name}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class MalariaCellImagesDataset(BaseVisionDataset):
    _registry_name = "malariacellimages"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "5458fc86f78a722584606080be6d07553350c2e1",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "c4278e23373e478d3f2cb7c38d0db373793174cd",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{self._registry_name}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_BINARY]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _BINARY


class Mit67Dataset(BaseVisionDataset):
    _registry_name = "mit67"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "b9c66c7dc304d78feac15f29246e02cc63a0c616",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "92fd40d9988360c70dcd70fa91ec2d0ce8b7adbf",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = (
                    self._INFO["local_dir"]
                    + f"{self._registry_name}/"
                    + f"{self._split}/"
                    + self._data[col].astype(str)
                )
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class NikeDataset(BaseVisionDataset):
    _registry_name = "nike"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "6c282d5269f2cf0e767a5c0149eba86375d93b15",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "d8ccb875ac1e3f79a3ce4adaff8f2f8ded1a7c63",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = (
                    self._INFO["local_dir"]
                    + f"{self._registry_name}/"
                    + f"{self._split}/"
                    + self._data[col].astype(str)
                )
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class NWPUResisc45Dataset(BaseVisionDataset):
    _registry_name = "nwpu_resisc45"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "3502dec1dcc373ef558960b80adc1482613c513c",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "781c45f506f7e677a24c7c0471464b86be469370",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{self._registry_name}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class OpenSurfacesMinc2500Dataset(BaseVisionDataset):
    _registry_name = "opensurfacesminc2500"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "7fb2f48959b52c65d720c44e1ff1202335fe6a66",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "a3b5d644ac4214c401ddf728cbe53ac1e2e5ec28",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{self._registry_name}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class OxfordFlowersDataset(BaseVisionDataset):
    _registry_name = "oxfordflowers"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "d8aaf75516370a517428efbc93f0356ca6023ce6",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "99e387e73371c460883caa5b120f43e2c851e347",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = (
                    self._INFO["local_dir"]
                    + f"{self._registry_name}/"
                    + f"{self._split}/"
                    + self._data[col].astype(str)
                )
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class RedfinDataset(BaseVisionDataset):
    _registry_name = "redfin"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "c6978f45296b58bfb291fea59ed0556356f844a9",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "abbdb74589a44869409c2e386adde7462a4fd900",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        logging.warning("The Redfin dataset contains classes that have only one training example.")
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{self._registry_name}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class SemartSchoolDataset(BaseVisionDataset):
    _registry_name = "semartschool"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "ddee19cdff1ac2d1a09c42b3508d8739fa7d59f1",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "9985f8a8ef840bc6ee3e99ec9d3844bbf06287fa",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = self._INFO["local_dir"] + f"{self._registry_name}/" + self._data[col].astype(str)
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class StanfordCarsDataset(BaseVisionDataset):
    _registry_name = "stanfordcars"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "cac116fec6a60569779f50f37a4d98ba06f44444",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "8157ba04823eb2f46f0dbbd54e32fd1b23f3d815",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = (
                    self._INFO["local_dir"]
                    + f"{self._registry_name}/"
                    + f"{self._split}/"
                    + self._data[col].astype(str)
                )
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS


class StanfordDogsDataset(BaseVisionDataset):
    _registry_name = "stanforddogs"
    _IMAGE_ZIP = _registry_name + ".zip"
    _ANNOTATION_ZIP = _registry_name + "_annotations.zip"
    _INFO = {
        "image": {
            "url": s3_repo_url + "vision_datasets/custom_labels_cv_datasets/image_classification/" + _IMAGE_ZIP,
            "sha1sum": "6284ef11ab7cd4118afe1dade8ecebac96d786e2",
        },
        "annotation": {
            "url": s3_repo_url
            + "vision_datasets/custom_labels_cv_datasets/image_classification/annotations/"
            + _ANNOTATION_ZIP,
            "sha1sum": "7bcf1bc38c22f83575605efbfb11ffc5ad208d46",
        },
        "local_dir": os.path.join(get_data_home_dir(), _registry_name) + "/",
    }

    def __init__(self, split="train"):
        self._split = split
        self._path = get_data_home_dir()
        self.image_zip = os.path.join(self._path, self._IMAGE_ZIP)
        self.annotation_zip = os.path.join(self._path, self._ANNOTATION_ZIP)
        # Download Images
        download(
            url=self._INFO["image"]["url"],
            path=self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
        )
        # Download Annotations
        download(
            url=self._INFO["annotation"]["url"],
            path=self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
        )
        protected_zip_extraction(
            self.image_zip,
            sha1_hash=self._INFO["image"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )
        protected_zip_extraction(
            self.annotation_zip,
            sha1_hash=self._INFO["annotation"]["sha1sum"],
            folder=self._INFO["local_dir"],
        )

        try:
            self._data = pd.read_csv(
                os.path.join(self._INFO["local_dir"], f"{self._registry_name}_{self._split}_annotations.csv")
            )
            _columns_to_drop = self._data.columns.difference(self.feature_columns + self.label_columns)
            self._data.drop(columns=_columns_to_drop, inplace=True)
            for col in self.feature_columns:
                self._data[col] = (
                    self._INFO["local_dir"]
                    + f"{self._registry_name}/"
                    + f"{self._split}/"
                    + self._data[col].astype(str)
                )
        except FileNotFoundError as e:
            logger.warn(f"Data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        return self._INFO["local_dir"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def data(self):
        return self._data

    @property
    def problem_type(self):
        return _MULTICLASS
