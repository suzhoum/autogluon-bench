import abc
import logging
import os

import pandas as pd

from autogluon.bench.utils.dataset_utils import get_data_home_dir, get_repo_url, path_expander
from autogluon.common.loaders import load_zip
from autogluon.common.loaders._utils import download

from .constants import (
    _BINARY,
    _CATEGORICAL,
    _IMAGE_SIMILARITY,
    _IMAGE_TEXT_SIMILARITY,
    _MULTICLASS,
    _NER,
    _NUMERICAL,
    _REGRESSION,
    _TEXT,
    _TEXT_SIMILARITY,
)

# Add dataset class names here
__all__ = [
    "Shopee",
    "StanfordOnline",
    "Flickr30k",
    "SNLI",
    "MitMovies",
    "WomenClothingReview",
    "MelBourneAirBnb",
    "AEPricePrediction",
    "IMDBGenrePrediction",
    "JCPennyCategory",
    "NewsPopularity",
    "NewsChannel",
]

logger = logging.getLogger(__name__)


class BaseMultiModalDataset(abc.ABC):
    def __init__(self, split: str, dataset_name: str, data_info: dict):
        """
        Initializes the class.

        Args:
            split (str): Specifies the dataset split. It should be one of the following options: 'train', 'val', 'test'.
        """
        try:
            ext = os.path.splitext(data_info[split]["url"])[-1]
            self._path = os.path.join(get_data_home_dir(), dataset_name, f"{split}{ext}")
            download(data_info[split]["url"], path=self._path)
            if ext == ".csv":
                self._data = pd.read_csv(self._path)
            elif ext == ".pq":
                self._data = pd.read_parquet(self._path)
            else:
                raise NotImplementedError(f"File extension {ext} is not supported.")
        except Exception:
            logger.warn(f"The data split {split} is not available.")
            self._data = None

        self._split = split

    @property
    @abc.abstractmethod
    def feature_columns(self):
        pass

    @property
    @abc.abstractmethod
    def label_columns(self):
        pass

    @property
    def label_types(self):
        pass

    @property
    @abc.abstractmethod
    def data(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def metric(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def problem_type(self):
        raise NotImplementedError


class BaseImageDataset(BaseMultiModalDataset):
    @property
    def base_folder(self):
        """Base folder that contains images"""
        pass

    @property
    @abc.abstractmethod
    def image_columns(self):
        """List of image columns"""
        pass


class BaseMatcherDataset(BaseMultiModalDataset):
    @property
    def match_label(self):
        """the label indicating that query and response have the same semantic meanings"""
        pass


class Shopee(BaseImageDataset):
    _SOURCE = ""
    _INFO = {
        "data": {
            "url": get_repo_url() + "vision_datasets/shopee.zip",
        },
    }
    _registry_name = "shopee"

    def __init__(self, split="train"):
        self._split = split
        self._path = os.path.join(get_data_home_dir(), "shopee")
        load_zip.unzip(self._INFO["data"]["url"], unzip_dir=self._path)
        self._base_folder = os.path.join(self._path, "shopee")
        try:
            data_path = os.path.join(self._base_folder, f"{self._split}.csv")
            self._data = pd.read_csv(data_path)
            self._data["image"] = self._data["image"].apply(
                lambda ele: path_expander(ele, base_folder=self._base_folder)
            )
        except FileNotFoundError as e:
            logger.warn(f"The data split {self._split} is not available.")
            self._data = None

    @property
    def base_folder(self):
        """Base folder that contains images"""
        return self._base_folder

    @property
    def image_columns(self):
        """List of image columns"""
        return ["image"]

    @property
    def data(self):
        return self._data

    @property
    def feature_columns(self):
        return ["image"]

    @property
    def label_columns(self):
        return ["label"]

    @property
    def metric(self):
        return "acc"

    @property
    def problem_type(self):
        return _MULTICLASS


class StanfordOnline(BaseMatcherDataset):
    _SOURCE = "https://cvgl.stanford.edu/projects/lifted_struct/"
    _INFO = {
        "data": {
            "url": get_repo_url() + "Stanford_Online_Products.zip",
        },
    }
    _registry_name = "stanford_online"

    def __init__(self, split="train"):
        self._split = split
        self._path = os.path.join(get_data_home_dir(), "Stanford_Online_Products")
        load_zip.unzip(self._INFO["data"]["url"], unzip_dir=self._path)
        self._base_folder = os.path.join(self._path, "Stanford_Online_Products")
        try:
            self._data = pd.read_csv(os.path.join(self._base_folder, f"{self._split}.csv"), index_col=0)
            self._image_columns = ["Image1", "Image2"]
            for image_col in self._image_columns:
                self._data[image_col] = self._data[image_col].apply(
                    lambda ele: path_expander(ele, base_folder=self._base_folder)
                )
        except FileNotFoundError as e:
            logger.warn(f"The data split {self._split} is not available.")
            self._data = None

    @property
    def image_columns(self):
        """List of image columns"""
        return self._image_columns

    @property
    def data(self):
        return self._data

    @property
    def feature_columns(self):
        return self._image_columns

    @property
    def label_columns(self):
        return ["Label"]

    @property
    def match_label(self):
        return 1

    @property
    def metric(self):
        return "roc_auc"

    @property
    def problem_type(self):
        return _IMAGE_SIMILARITY


class Flickr30k(BaseMatcherDataset):
    _SOURCE = "https://paperswithcode.com/dataset/flickr30k"
    _INFO = {
        "data": {"url": get_repo_url() + "flickr30k.zip"},
    }
    _registry_name = "flickr30k"

    def __init__(self, split="train"):
        self._split = split
        self._path = os.path.join(get_data_home_dir(), "flickr30k")
        load_zip.unzip(self._INFO["data"]["url"], unzip_dir=self._path)
        self._base_folder = os.path.join(self._path, "flickr30k_processed")

        try:
            self._data = pd.read_csv(os.path.join(self._base_folder, f"{self._split}.csv"), index_col=0)
            self._image_col = "image"
            self._text_col = "caption"

            self._data[self._image_col] = self._data[self._image_col].apply(
                lambda ele: path_expander(ele, base_folder=self._base_folder)
            )
            self._label_col = "relevance"
            self._data[self._label_col] = [1] * len(self._data)
        except FileNotFoundError as e:
            logger.warn(f"The data split {self._split} is not available.")
            self._data = None

    @property
    def image_columns(self):
        """List of image columns"""
        return [self._image_col]

    @property
    def text_columns(self):
        """List of text columns"""
        return [self._text_col]

    @property
    def data(self):
        return self._data

    @property
    def feature_columns(self):
        return [self._image_col, self._text_col]

    @property
    def label_columns(self):
        return [self._label_col]

    @property
    def match_label(self):
        return 1

    @property
    def metric(self):
        return "recall"

    @property
    def problem_type(self):
        return _IMAGE_TEXT_SIMILARITY


class SNLI(BaseMatcherDataset):
    _SOURCE = "https://nlp.stanford.edu/projects/snli/"
    _INFO = {
        "train": {
            "url": get_repo_url() + "snli/snli_train.csv",
        },
        "test": {"url": get_repo_url() + "snli/snli_test.csv"},
    }
    _registry_name = "snli"

    def __init__(self, split="train"):
        self._split = split
        self._path = os.path.join(get_data_home_dir(), "snli", f"{split}.csv")
        try:
            download(self._INFO[split]["url"], path=self._path)
            self._data = pd.read_csv(self._path, delimiter="|")
        except Exception:
            logger.warn(f"The data split {self._split} is not available.")
            self._data = None

    @property
    def text_columns(self):
        """List of text columns"""
        return ["premise", "hypothesis"]

    @property
    def data(self):
        return self._data

    @property
    def feature_columns(self):
        return ["premise", "hypothesis"]

    @property
    def label_columns(self):
        return ["label"]

    @property
    def match_label(self):
        return 1

    @property
    def metric(self):
        return "roc_auc"

    @property
    def problem_type(self):
        return _TEXT_SIMILARITY


class MitMovies(BaseMultiModalDataset):
    _SOURCE = "https://groups.csail.mit.edu/sls/downloads/movie/"
    _INFO = {
        "train": {
            "url": get_repo_url() + "ner/mit-movies/train_v2.csv",
        },
        "test": {
            "url": get_repo_url() + "ner/mit-movies/test_v2.csv",
        },
    }
    _registry_name = "mit_movies"

    def __init__(self, split="train"):
        super().__init__(split=split, dataset_name=self._registry_name, data_info=self._INFO)

    @property
    def feature_columns(self):
        return ["text_snippet"]

    @property
    def label_columns(self):
        return ["entity_annotations"]

    @property
    def metric(self):
        return ["overall_recall", "overall_precision", "overall_f1", "actor"]

    @property
    def problem_type(self):
        return _NER

    @property
    def data(self):
        return self._data


class WomenClothingReview(BaseMultiModalDataset):
    _SOURCE = "https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews"
    _INFO = {
        "train": {
            "url": get_repo_url() + "women_clothing_review/train.pq",
        },
        "test": {
            "url": get_repo_url() + "women_clothing_review/test.pq",
        },
    }
    _registry_name = "women_clothing_review"

    def __init__(self, split="train"):
        super().__init__(split=split, dataset_name=self._registry_name, data_info=self._INFO)

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def feature_columns(self):
        return ["Title", "Review Text", "Age", "Division Name", "Department Name", "Class Name"]

    @property
    def feature_types(self):
        return [_TEXT, _TEXT, _NUMERICAL, _CATEGORICAL, _CATEGORICAL, _CATEGORICAL]

    @property
    def fill_na_value(self):
        """The default function to fill missing values"""
        return {"Division Name": "None", "Department Name": "None", "Class Name": "None"}

    @property
    def label_columns(self):
        return ["Rating"]

    @property
    def label_types(self):
        return [_NUMERICAL]

    @property
    def data(self):
        return self._data

    @property
    def metric(self):
        return "r2"

    @property
    def problem_type(self):
        return _REGRESSION


class MelBourneAirBnb(BaseMultiModalDataset):
    _SOURCE = "https://www.kaggle.com/tylerx/melbourne-airbnb-open-data"
    _INFO = {
        "train": {
            "url": get_repo_url() + "airbnb_melbourne/train.pq",
        },
        "test": {
            "url": get_repo_url() + "airbnb_melbourne/test.pq",
        },
    }
    _registry_name = "melbourne_airbnb"

    def __init__(self, split="train"):
        super().__init__(split=split, dataset_name=self._registry_name, data_info=self._INFO)

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def ignore_columns(self):
        return [
            "id",
            "listing_url",
            "scrape_id",
            "last_scraped",
            "picture_url",
            "host_id",
            "host_url",
            "host_name",
            "host_thumbnail_url",
            "host_picture_url",
            "monthly_price",
            "weekly_price",
            "price",
            "calendar_last_scraped",
        ]

    @property
    def label_columns(self):
        return ["price_label"]

    @property
    def feature_columns(self):
        all_columns = sorted(self._data.columns)
        feature_columns = [
            col for col in all_columns if col not in self.label_columns and col not in self.ignore_columns
        ]
        return feature_columns

    @property
    def data(self):
        return self._data

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def metric(self):
        return "acc"

    @property
    def problem_type(self):
        return _MULTICLASS


class AEPricePrediction(BaseMultiModalDataset):
    _SOURCE = "https://www.kaggle.com/PromptCloudHQ/innerwear-data-from-victorias-secret-and-others"
    _INFO = {
        "train": {
            "url": get_repo_url() + "ae_price_prediction/train.pq",
        },
        "test": {
            "url": get_repo_url() + "ae_price_prediction/test.pq",
        },
    }
    _registry_name = "ae_price_prediction"

    def __init__(self, split="train"):
        super().__init__(split=split, dataset_name=self._registry_name, data_info=self._INFO)

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def data(self):
        return self._data

    @property
    def ignore_columns(self):
        return ["mrp", "pdp_url"]

    @property
    def feature_columns(self):
        return [col for col in self.data.columns if col not in self.ignore_columns and col not in self.label_columns]

    @property
    def label_columns(self):
        return ["price"]

    @property
    def label_types(self):
        return [_NUMERICAL]

    @property
    def metric(self):
        return "r2"

    @property
    def problem_type(self):
        return _REGRESSION


class IMDBGenrePrediction(BaseMultiModalDataset):
    _SOURCE = "https://www.kaggle.com/PromptCloudHQ/imdb-data"
    _INFO = {
        "train": {
            "url": get_repo_url() + "imdb_genre_prediction/train.csv",
        },
        "test": {
            "url": get_repo_url() + "imdb_genre_prediction/test.csv",
        },
    }
    _registry_name = "imdb_genre_prediction"

    def __init__(self, split="train"):
        super().__init__(split=split, dataset_name=self._registry_name, data_info=self._INFO)

    @property
    def data(self):
        return self._data

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def data(self):
        return self._data

    @property
    def label_columns(self):
        return ["Genre_is_Drama"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def feature_columns(self):
        return [col for col in list(self.data.columns) if col not in self.label_columns]

    @property
    def metric(self):
        return "roc_auc"

    @property
    def problem_type(self):
        return _BINARY


class JCPennyCategory(BaseMultiModalDataset):
    _SOURCE = "https://www.kaggle.com/PromptCloudHQ/all-jc-penny-products"
    _INFO = {
        "train": {
            "url": get_repo_url() + "jc_penney_products/train.csv",
        },
        "test": {
            "url": get_repo_url() + "jc_penney_products/test.csv",
        },
    }
    _registry_name = "jc_penney_products"

    def __init__(self, split="train"):
        super().__init__(split=split, dataset_name=self._registry_name, data_info=self._INFO)

    @property
    def data(self):
        return self._data

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def data(self):
        return self._data

    @property
    def label_columns(self):
        return ["sale_price"]

    @property
    def label_types(self):
        return [_NUMERICAL]

    @property
    def feature_columns(self):
        return [col for col in list(self.data.columns) if col not in self.label_columns]

    @property
    def metric(self):
        return "r2"

    @property
    def problem_type(self):
        return _REGRESSION


class NewsPopularity(BaseMultiModalDataset):
    _SOURCE = "https://archive.ics.uci.edu/ml/datasets/online+news+popularity"
    _INFO = {
        "train": {
            "url": get_repo_url() + "news_popularity2/train.csv",
        },
        "test": {
            "url": get_repo_url() + "news_popularity2/test.csv",
        },
    }
    _registry_name = "news_popularity"

    def __init__(self, split="train"):
        super().__init__(split=split, dataset_name=self._registry_name, data_info=self._INFO)

    @property
    def data(self):
        return self._data

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def data(self):
        return self._data

    @property
    def label_columns(self):
        return ["log_shares"]

    @property
    def label_types(self):
        return [_NUMERICAL]

    @property
    def feature_columns(self):
        return [col for col in list(self.data.columns) if col not in self.label_columns]

    @property
    def metric(self):
        return "r2"

    @property
    def problem_type(self):
        return _REGRESSION


class NewsChannel(BaseMultiModalDataset):
    _SOURCE = "https://archive.ics.uci.edu/ml/datasets/online+news+popularity"
    _INFO = {
        "train": {
            "url": get_repo_url() + "news_channel/train.csv",
        },
        "test": {
            "url": get_repo_url() + "news_channel/test.csv",
        },
    }
    _registry_name = "news_channel"

    def __init__(self, split="train"):
        super().__init__(split=split, dataset_name=self._registry_name, data_info=self._INFO)

    @property
    def data(self):
        return self._data

    @classmethod
    def splits(cls):
        return cls._INFO.keys()

    @property
    def data(self):
        return self._data

    @property
    def label_columns(self):
        return ["channel"]

    @property
    def label_types(self):
        return [_CATEGORICAL]

    @property
    def feature_columns(self):
        return [col for col in list(self.data.columns) if col not in self.label_columns]

    @property
    def metric(self):
        return "acc"

    @property
    def problem_type(self):
        return _MULTICLASS
