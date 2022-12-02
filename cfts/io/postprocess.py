import hashlib
import shutil
import os
from pathlib import Path
import sys
import zipfile

from tqdm import tqdm

from psi import get_config


def zip_data():
    import argparse
    parser = argparse.ArgumentParser('cfts-zip-data')
    parser.add_argument('path', nargs='?', default=get_config('DATA_ROOT'), type=Path)
    parser.add_argument('-d', '--destination', type=Path)
    args = parser.parse_args()

    dirs = [p for p in args.path.iterdir() if p.is_dir()]
    for path in tqdm(dirs):
        shutil.make_archive(str(path), 'zip', str(path))
        try:
            zippath = validate(path)
            zipmd5 = md5sum(zippath.open('rb'))
            md5path = zippath.with_suffix('.md5')
            md5path.write_text(zipmd5)
            shutil.rmtree(path)
            if args.destination is not None:
                for file in (zippath, md5path):
                    new_file = args.destination / file.name
                    file.rename(new_file)
        except IOError as e:
            print(e)


def md5sum(stream, blocksize=1024**2):
    '''
    Generates md5sum from byte stream

    Parameters
    ----------
    stream : stream
        Any object supporting a `read` method that returns bytes.
    blocksize : int
        Blocksize to use for computing md5sum

    Returns
    -------
    md5sum : str
        Hexdigest of md5sum for stream
    '''
    md5 = hashlib.md5()
    while True:
        block = stream.read(blocksize)
        if not block:
            break
        md5.update(block)
    return md5.hexdigest()


def validate(path):
    '''
    Validates contents of zipfile using md5sum

    Parameters
    ----------
    path : {str, pathlib.Path}
        Path containing data that was zipped. Zipfile is expected to have the
        same path, but ending in ".zip".

    The zipfile is opened and iterated through. The MD5 sum for each file
    inside the archive is compared with the companion file in the unzipped
    folder.
    '''
    zippath = Path(path).with_suffix('.zip')
    archive = zipfile.ZipFile(zippath)
    for name in archive.namelist():
        archive_md5 = md5sum(archive.open(name))
        file = path / name
        if file.is_file():
            with file.open('rb') as fh:
                file_md5 = md5sum(fh)
            if archive_md5 != file_md5:
                raise IOError('{name} in zipfile for {path} is corrupted')
    return zippath
