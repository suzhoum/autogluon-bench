import functools
import hashlib
import os
import sys
import uuid
import warnings
from typing import Optional

import boto3
import numpy as np
import requests
import tqdm

S3_PREFIX = "s3://"

INT_TYPES = (int, np.uint8, np.int8, np.int32, np.int64)
FLOAT_TYPES = (float, np.float16, np.float32, np.float64)
BOOL_TYPES = (bool, np.bool_)


if not sys.platform.startswith("win32"):
    # refer to https://github.com/untitaker/python-atomicwrites
    def replace_file(src, dst):
        """Implement atomic os.replace with linux and OSX.
        Parameters
        ----------
        src : source file path
        dst : destination file path
        """
        try:
            os.rename(src, dst)
        except OSError:
            try:
                os.remove(src)
            except OSError:
                pass
            finally:
                raise OSError(
                    "Moving downloaded temp file - {}, to {} failed. \
                    Please retry the download.".format(
                        src, dst
                    )
                )

else:
    import ctypes

    _MOVEFILE_REPLACE_EXISTING = 0x1
    # Setting this value guarantees that a move performed as a copy
    # and delete operation is flushed to disk before the function returns.
    # The flush occurs at the end of the copy operation.
    _MOVEFILE_WRITE_THROUGH = 0x8
    _windows_default_flags = _MOVEFILE_WRITE_THROUGH

    def _str_to_unicode(x):
        """Handle text decoding. Internal use only"""
        if not isinstance(x, str):
            return x.decode(sys.getfilesystemencoding())
        return x

    def _handle_errors(rv, src):
        """Handle WinError. Internal use only"""
        if not rv:
            msg = ctypes.FormatError(ctypes.GetLastError())
            # if the MoveFileExW fails(e.g. fail to acquire file lock), removes the tempfile
            try:
                os.remove(src)
            except OSError:
                pass
            finally:
                raise OSError(msg)

    def replace_file(src, dst):
        """Implement atomic os.replace with windows.
        refer to https://docs.microsoft.com/en-us/windows/desktop/api/winbase/nf-winbase-movefileexw
        The function fails when one of the process(copy, flush, delete) fails.
        Parameters
        ----------
        src : source file path
        dst : destination file path
        """
        _handle_errors(
            ctypes.windll.kernel32.MoveFileExW(
                _str_to_unicode(src), _str_to_unicode(dst), _windows_default_flags | _MOVEFILE_REPLACE_EXISTING
            ),
            src,
        )


def sha1sum(filename):
    """Calculate the sha1sum of a file
    Parameters
    ----------
    filename
        Name of the file
    Returns
    -------
    ret
        The sha1sum
    """
    with open(filename, mode="rb") as f:
        d = hashlib.sha1()
        for buf in iter(functools.partial(f.read, 1024 * 100), b""):
            d.update(buf)
    return d.hexdigest()


def download(
    url: str,
    path: Optional[str] = None,
    overwrite: Optional[bool] = False,
    sha1_hash: Optional[str] = None,
    retries: Optional[int] = 5,
    verify_ssl: Optional[bool] = True,
) -> str:
    """Download a given URL

    Parameters
    ----------
    url
        URL to download
    path
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite
        Whether to overwrite destination file if already exists.
    sha1_hash
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries
        The number of times to attempt the download in case of failure or non 200 return codes
    verify_ssl
        Verify SSL certificates.
    Returns
    -------
    fname
        The file path of the downloaded file.
    """
    is_s3 = url.startswith(S3_PREFIX)
    if is_s3:
        s3 = boto3.resource("s3")
        if boto3.session.Session().get_credentials() is None:
            from botocore.handlers import disable_signing

            s3.meta.client.meta.events.register("choose-signer.s3.*", disable_signing)
        components = url[len(S3_PREFIX) :].split("/")
        if len(components) < 2:
            raise ValueError("Invalid S3 url. Received url={}".format(url))
        s3_bucket_name = components[0]
        s3_key = "/".join(components[1:])
    if path is None:
        fname = url.split("/")[-1]
        # Empty filenames are invalid
        assert fname, "Can't construct file-name from this URL. " "Please set the `path` option manually."
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split("/")[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0, currently it's {}".format(retries)

    if not verify_ssl:
        warnings.warn(
            "Unverified HTTPS request is being made (verify_ssl=False). "
            "Adding certificate verification is strongly advised."
        )

    print("~~~~~~~~~~~~sha1sum", sha1sum(fname))
    if overwrite or not os.path.exists(fname) or (sha1_hash and not sha1sum(fname) == sha1_hash):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                print("Downloading {} from {}...".format(fname, url))
                if is_s3:
                    response = s3.meta.client.head_object(Bucket=s3_bucket_name, Key=s3_key)
                    total_size = int(response.get("ContentLength", 0))
                    random_uuid = str(uuid.uuid4())
                    tmp_path = "{}.{}".format(fname, random_uuid)
                    if tqdm is not None:

                        def hook(t_obj):
                            def inner(bytes_amount):
                                t_obj.update(bytes_amount)

                            return inner

                        with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True) as t:
                            s3.meta.client.download_file(s3_bucket_name, s3_key, tmp_path, Callback=hook(t))
                    else:
                        s3.meta.client.download_file(s3_bucket_name, s3_key, tmp_path)
                else:
                    r = requests.get(url, stream=True, verify=verify_ssl)
                    if r.status_code != 200:
                        raise RuntimeError("Failed downloading url {}".format(url))
                    # create uuid for temporary files
                    random_uuid = str(uuid.uuid4())
                    total_size = int(r.headers.get("content-length", 0))
                    chunk_size = 1024
                    if tqdm is not None:
                        t = tqdm.tqdm(total=total_size, unit="iB", unit_scale=True)
                    with open("{}.{}".format(fname, random_uuid), "wb") as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:  # filter out keep-alive new chunks
                                if tqdm is not None:
                                    t.update(len(chunk))
                                f.write(chunk)
                    if tqdm is not None:
                        t.close()
                # if the target file exists(created by other processes)
                # and have the same hash with target file
                # delete the temporary file
                if not os.path.exists(fname) or (sha1_hash and not sha1sum(fname) == sha1_hash):
                    # atomic operation in the same file system
                    replace_file("{}.{}".format(fname, random_uuid), fname)
                else:
                    try:
                        os.remove("{}.{}".format(fname, random_uuid))
                    except OSError:
                        pass
                    finally:
                        warnings.warn("File {} exists in file system so the downloaded file is deleted".format(fname))
                if sha1_hash and not sha1sum(fname) == sha1_hash:
                    raise UserWarning(
                        "File {} is downloaded but the content hash does not match."
                        " The repo may be outdated or download may be incomplete. "
                        'If the "repo_url" is overridden, consider switching to '
                        "the default repo.".format(fname)
                    )
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e

                print(
                    "download failed due to {}, retrying, {} attempt{} left".format(
                        repr(e), retries, "s" if retries > 1 else ""
                    )
                )

    return fname


def get_home_dir():
    """Get home directory"""
    _home_dir = os.environ.get("AUTO_MM_BENCH_HOME", os.path.join("~", ".auto_mm_bench"))
    # expand ~ to actual path
    _home_dir = os.path.expanduser(_home_dir)
    return _home_dir


def get_data_home_dir():
    """Get home directory for storing the datasets"""
    home_dir = get_home_dir()
    return os.path.join(home_dir, "datasets")


def get_repo_url():
    """Return the base URL for dataset repository"""
    default_repo = "https://automl-mm-bench.s3.amazonaws.com"
    repo_url = os.environ.get("AUTO_MM_BENCH_REPO", default_repo)
    if repo_url[-1] != "/":
        repo_url = repo_url + "/"
    return repo_url
