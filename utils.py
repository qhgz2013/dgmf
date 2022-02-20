import re
from typing import *
import logging
import sys
import random
import numpy
import torch
import os
import shutil
from time import time

logger = logging.getLogger('util')


def configure_logging(verbose: bool = False) -> logging.Logger:
    class ScreenOutputFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> int:
            return record.levelno in (logging.DEBUG, logging.INFO)
    fmt_str = '[%(asctime)s] [%(levelname)s] [%(name)s] (%(filename)s:%(lineno)d) %(message)s'
    fmt = logging.Formatter(fmt_str)
    root = logging.root

    # ROOT logging config
    stdout_handler = logging.StreamHandler(sys.stdout)
    if verbose:
        stdout_handler.setLevel(logging.DEBUG)
    else:
        stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(fmt)
    stdout_handler.addFilter(ScreenOutputFilter())
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(fmt)
    root.addHandler(stdout_handler)
    root.addHandler(stderr_handler)
    root.setLevel(logging.DEBUG)
    return root


def fix_seed(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_random_state() -> List[Any]:
    states = [random.getstate(), numpy.random.get_state(), torch.random.get_rng_state(), torch.cuda.get_rng_state_all()]
    return states


def set_random_state(state: List[Any]):
    random.setstate(state[0])
    numpy.random.set_state(state[1])
    torch.random.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


def set_target_gpu(gpu: str):
    logger.debug('Setting gpu: %s', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


def reset_dir(target_dir: str):
    if os.path.isdir(target_dir):
        logger.debug('Removing dir: %s', target_dir)
        shutil.rmtree(target_dir)
    logger.debug('Creating dir: %s', target_dir)
    os.makedirs(target_dir)


def format_size(size: int) -> str:
    if size < 2 ** 10:
        return f'{size}B'
    if size < 2 ** 20:
        return f'{format(size / (2 ** 10), ".3f")}kB'
    if size < 2 ** 30:
        return f'{format(size / (2 ** 20), ".3f")}MB'
    return f'{format(size / (3 ** 30), ".3f")}GB'


def _list_dir_by_mtime(path: str, ignore_file_regex: Optional[str] = None) -> List[str]:
    assert os.path.isdir(path), '"%s" is not a directory' % path
    files = os.listdir(path)
    logger.debug('%d files found in directory "%s"', len(files), path)
    if len(files) == 0:
        return []
    ignore_file_regex = re.compile(ignore_file_regex) if ignore_file_regex is not None else None
    file_stats = {file: os.stat(os.path.join(path, file)) for file in files
                  if ignore_file_regex is None or re.search(ignore_file_regex, file) is None}
    file_sorted = sorted(file_stats.items(), key=lambda x: x[1].st_mtime, reverse=True)  # latest first
    return [x[0] for x in file_sorted]


def save_model(target_path: str, params: Any, epoch: int, include_random_state: bool = True,
               max_keep_weights: Optional[int] = None, ignore_file_regex: Optional[str] = None):
    parent_dir = os.path.abspath(os.path.join(target_path, '..'))
    os.makedirs(parent_dir, exist_ok=True)
    logger.debug('Saving model to "%s" at epoch %d', target_path, epoch)
    obj = {
        'params': params,
        'epoch': epoch,
    }
    if include_random_state:
        obj['rng_state'] = get_random_state()
    t1 = time()
    torch.save(obj, target_path)
    t2 = time()
    obj_stat = os.stat(target_path)
    logger.debug('Model saved, time used: %f, file size: %s', t2 - t1, format_size(obj_stat.st_size))
    if max_keep_weights is not None and max_keep_weights > 0:
        files = _list_dir_by_mtime(os.path.abspath(os.path.join(target_path, '..')), ignore_file_regex)
        if len(files) > max_keep_weights:
            files_to_remove = files[max_keep_weights:]
            for file in files_to_remove:
                file_path = os.path.abspath(os.path.join(target_path, '..', file))
                logger.debug('Removing file "%s" due to max_keep_weights exceeded', file_path)
                os.remove(file_path)


def load_model(target_path: str, use_saved_random_state: bool = True) -> Tuple[int, Any]:
    logger.debug('Load model from "%s"', target_path)
    t1 = time()
    obj = torch.load(target_path)
    t2 = time()
    logger.debug('Loaded model, time used: %f, saved epoch: %d', t2 - t1, obj['epoch'])
    if use_saved_random_state and 'rng_state' in obj:
        logger.debug('Setting random state')
        set_random_state(obj['rng_state'])
    return obj['epoch'], obj['params']


def load_last_saved_model(target_path: str, use_saved_random_state: bool = True,
                          ignore_file_regex: Optional[str] = None) -> Optional[Tuple[int, Any]]:
    logger.debug('Load last saved model from "%s"', target_path)
    if not os.path.exists(target_path):
        return None
    if os.path.isfile(target_path):
        return load_model(target_path, use_saved_random_state)
    file_selected = _list_dir_by_mtime(target_path, ignore_file_regex)[0]
    v = load_model(target_path=os.path.join(target_path, file_selected), use_saved_random_state=use_saved_random_state)
    return v
