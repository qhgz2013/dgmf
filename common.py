import argparse
import torch
import numpy as np
import scipy.sparse as sp
from typing import *
import utils
from tqdm import tqdm
import dataset_loader
from functools import partial
import os
import torch.utils.tensorboard as tb
from datetime import datetime
import shutil
import sys


dataset_mapper = {
    'gowalla': dataset_loader.GowallaDataset,
    'foursquare': dataset_loader.FoursquareDataset
}


def l2_loss(factor: Union[float, torch.FloatTensor], *params: torch.Tensor) -> torch.Tensor:
    # noinspection PyTypeChecker
    return torch.sum(torch.stack([torch.sum(var ** 2) for var in params])) * factor / 2


def split_train_test(sp_interaction_mat: sp.spmatrix, ratio: float = 0.9) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    sp_interaction_mat = sp_interaction_mat.tocsr()
    if ratio == 1.0:
        return sp_interaction_mat, sp_interaction_mat
    train_sp_mat_indptr = np.zeros_like(sp_interaction_mat.indptr)
    test_sp_mat_indptr = np.zeros_like(sp_interaction_mat.indptr)
    n_pois = sp_interaction_mat.indptr[1:] - sp_interaction_mat.indptr[:-1]
    train_sp_mat_indices = []
    test_sp_mat_indices = []
    train_sp_mat_data = []
    test_sp_mat_data = []
    for i in range(sp_interaction_mat.indptr.size - 1):
        train_pois = int(ratio * n_pois[i])
        test_pois = n_pois[i] - train_pois
        idx = np.arange(n_pois[i], dtype=np.int32)
        np.random.shuffle(idx)
        data = sp_interaction_mat.data[sp_interaction_mat.indptr[i]:sp_interaction_mat.indptr[i+1]]
        indices = sp_interaction_mat.indices[sp_interaction_mat.indptr[i]:sp_interaction_mat.indptr[i+1]]
        train_sp_mat_indices.extend(indices[idx[:train_pois]])
        train_sp_mat_data.extend(data[idx[:train_pois]])
        test_sp_mat_indices.extend(indices[idx[train_pois:]])
        test_sp_mat_data.extend(data[idx[train_pois:]])
        train_sp_mat_indptr[i+1] = train_sp_mat_indptr[i] + train_pois
        test_sp_mat_indptr[i+1] = test_sp_mat_indptr[i] + test_pois
    train_sp_mat_indices = np.array(train_sp_mat_indices, dtype=np.int32)
    train_sp_mat_data = np.array(train_sp_mat_data, dtype=np.int32)
    test_sp_mat_indices = np.array(test_sp_mat_indices, dtype=np.int32)
    test_sp_mat_data = np.array(test_sp_mat_data, dtype=np.int32)
    train_sp_mat = sp.csr_matrix((train_sp_mat_data, train_sp_mat_indices, train_sp_mat_indptr),
                                 shape=sp_interaction_mat.shape)
    test_sp_mat = sp.csr_matrix((test_sp_mat_data, test_sp_mat_indices, test_sp_mat_indptr),
                                shape=sp_interaction_mat.shape)
    return train_sp_mat, test_sp_mat


def neg_sampling(interaction_id: Set[int], user: int, n_pois: int, neg_size: int) -> List[int]:
    samples = []
    while len(samples) < neg_size:
        sampled_pois = np.random.choice(n_pois, neg_size, replace=False)
        poi_ids = user * n_pois + sampled_pois
        for poi, poi_id in zip(sampled_pois, poi_ids):
            if poi_id not in interaction_id:
                samples.append(poi)
            if len(samples) == neg_size:
                break
    return samples


def fix_test_samples(test_sp_int_mat: sp.csr_matrix, interaction_id: Set[int], neg_size: int = 100)\
        -> Tuple[np.ndarray, np.ndarray]:
    n_users, n_pois = test_sp_int_mat.shape
    sampled_pois = np.empty((n_users, neg_size), dtype=np.int32)
    sampled_label = np.zeros_like(sampled_pois)
    for u in range(n_users):
        pos = test_sp_int_mat.indices[test_sp_int_mat.indptr[u]:test_sp_int_mat.indptr[u + 1]]
        if len(pos) >= neg_size:
            sampled_pois[u, :] = pos[:neg_size]
            sampled_label[u, :] = 1
            continue  # warn
        sampled_pois[u, :len(pos)] = pos
        neg = neg_sampling(interaction_id, u, n_pois, neg_size - len(pos))
        sampled_pois[u, len(pos):] = neg
        sampled_label[u, :len(pos)] = 1
    return sampled_pois, sampled_label


def build_interaction_id(sp_int_matrix: Union[sp.csr_matrix, sp.coo_matrix]) -> Set[int]:
    n_pois = sp_int_matrix.shape[1]
    if sp.isspmatrix_coo(sp_int_matrix):
        return set(sp_int_matrix.row * n_pois + sp_int_matrix.col)
    elif sp.isspmatrix_csr(sp_int_matrix):
        ret = set()
        indices = sp_int_matrix.indices
        for u in range(sp_int_matrix.shape[0]):
            ret.update(u * n_pois + indices[sp_int_matrix.indptr[u]:sp_int_matrix.indptr[u+1]])
        return ret
    raise TypeError(f'Unexpected type: {type(sp_int_matrix)}')


def interaction_iterator(sp_int_matrix: Union[sp.csr_matrix, sp.coo_matrix], interaction_id: Set[int], neg_size: int,
                         yield_data: bool = False, neg_weight: float = 0.0,
                         tqdm_kwargs: Optional[Dict[str, Any]] = None, pos_weight: float = 1.0) \
        -> Iterator[Tuple[int, int, float]]:
    if sp.isspmatrix_csr(sp_int_matrix):
        sp_int_matrix = sp_int_matrix.tocoo()
    if isinstance(neg_weight, int):
        neg_weight = float(neg_weight)
    if isinstance(pos_weight, int):
        pos_weight = float(pos_weight)
    users, pois, data = sp_int_matrix.row, sp_int_matrix.col, sp_int_matrix.data
    idx = np.arange(len(users), dtype=np.int32)
    np.random.shuffle(idx)
    users, pois, data = users[idx], pois[idx], data[idx]
    tqdm_kwargs = {} if tqdm_kwargs is None else tqdm_kwargs
    for i in tqdm(range(len(users)), **tqdm_kwargs):
        yield users[i], pois[i], (data[i] if yield_data else pos_weight)
        for neg_poi in neg_sampling(interaction_id, users[i], sp_int_matrix.shape[1], neg_size):
            yield users[i], neg_poi, neg_weight


def pairwise_interaction_iterator(sp_int_matrix: Union[sp.csr_matrix, sp.coo_matrix], interaction_id: Set[int],
                                  neg_size: int, yield_data: bool = False, neg_weight: float = 0,
                                  tqdm_kwargs: Optional[Dict[str, Any]] = None, max_pair_per_interaction: int = 5,
                                  pos_weight: float = 1.0) \
        -> Iterator[Tuple[int, int, int, float]]:
    if sp.isspmatrix_coo(sp_int_matrix):
        sp_int_matrix = sp_int_matrix.tocsr()
    if isinstance(neg_weight, int):
        neg_weight = float(neg_weight)
    if isinstance(pos_weight, int):
        pos_weight = float(pos_weight)
    default_diff = pos_weight - neg_weight  # if yield_data set to false, return this
    indices, indptr, data = sp_int_matrix.indices, sp_int_matrix.indptr, sp_int_matrix.data
    users = np.arange(sp_int_matrix.shape[0], dtype=np.int32)
    np.random.shuffle(users)
    for u in tqdm(users, **tqdm_kwargs):
        pois = indices[indptr[u]:indptr[u+1]]
        checkins = data[indptr[u]:indptr[u+1]]
        length = len(pois)
        if length <= max_pair_per_interaction + 1:
            for i1 in range(length):
                # positive-positive pairs
                for i2 in range(i1 + 1, length):
                    diff = checkins[i1] - checkins[i2]
                    yield u, pois[i1], pois[i2], (diff if yield_data else np.sign(diff) * default_diff)
                # positive-negative pairs
                for i2 in neg_sampling(interaction_id, u, sp_int_matrix.shape[1], neg_size):
                    yield u, pois[i1], i2, checkins[i1] if yield_data else default_diff
        else:
            for i1 in range(length):
                for i2 in np.random.choice(length - 1, max_pair_per_interaction, replace=False):
                    if checkins[i1] == checkins[i2]:
                        continue
                    i2 = i2 + 1 if i2 >= i1 else i2
                    diff = checkins[i1] - checkins[i2]
                    yield u, pois[i1], pois[i2], (diff if yield_data else np.sign(diff) * default_diff)
                for i2 in neg_sampling(interaction_id, u, sp_int_matrix.shape[1], neg_size):
                    yield u, pois[i1], i2, checkins[i1] if yield_data else default_diff


def batch_iterator(iterator_func: Callable[[], Iterator[tuple]], batch_size: int) -> Iterator[tuple]:
    itr = iter(iterator_func())
    first_iter = next(itr)
    n_args = len(first_iter)
    batch_data = []
    for arg_idx in range(n_args):
        if hasattr(first_iter[arg_idx], 'dtype') and isinstance(first_iter[arg_idx].dtype, np.dtype):
            # infer dtype from return value
            batch_data.append(np.empty(batch_size, dtype=first_iter[arg_idx].dtype))
        elif isinstance(first_iter[arg_idx], int):
            batch_data.append(np.empty(batch_size, dtype=np.int32))  # int -> int32
        elif isinstance(first_iter[arg_idx], float):
            batch_data.append(np.empty(batch_size, dtype=np.float32))  # float -> float32
        else:
            raise ValueError(f'Invalid scalar type: {type(first_iter[arg_idx])}')
        batch_data[arg_idx][0] = first_iter[arg_idx]
    batch_idx = 1
    for args in itr:
        if batch_idx == batch_size:
            yield tuple(batch_data)
            batch_idx = 0
        for arg_idx in range(n_args):
            batch_data[arg_idx][batch_idx] = args[arg_idx]
        batch_idx += 1
    if batch_idx > 0:
        yield tuple(x[:batch_idx] for x in batch_data)


def _unique_internal(func_internal: Callable, a: torch.Tensor, return_inverse: bool = False,
                     return_counts: bool = False, return_index: bool = False):
    if not return_index:
        return func_internal(a, return_inverse=return_inverse, return_counts=return_counts)
    v = func_internal(a, return_inverse=True, return_counts=return_counts)
    unique_a = v[0]
    inv_idx = v[1]
    idx = torch.empty_like(unique_a, dtype=inv_idx.dtype, device=inv_idx.device)
    perm = torch.arange(inv_idx.size(0), dtype=inv_idx.dtype, device=inv_idx.device)
    idx[inv_idx] = perm  # gather op
    if return_inverse:
        if return_counts:
            return unique_a, inv_idx, v[2], idx
        return unique_a, inv_idx, idx
    if return_counts:
        return unique_a, v[2], idx
    return unique_a, idx


unique = partial(_unique_internal, torch.unique)
unique_consecutive = partial(_unique_internal, torch.unique_consecutive)
summary_writer = None  # type: Optional[tb.SummaryWriter]


def pr_compute(gt_label_in_pred_ranking: torch.Tensor, ks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if gt_label_in_pred_ranking.dtype != torch.bool:
        gt_label_in_pred_ranking = torch.gt(gt_label_in_pred_ranking, 0)
    n_gt_pos = torch.unsqueeze(torch.sum(gt_label_in_pred_ranking, -1), 0)  # [1, n_users]
    tp = torch.empty((ks.size(0), gt_label_in_pred_ranking.size(0)), dtype=torch.float,
                     device=gt_label_in_pred_ranking.device)  # [n_ks, n_users]
    for i, k in enumerate(torch.split(ks, 1)):
        tp[i] = torch.sum(gt_label_in_pred_ranking[:, :k], -1)
    pre = torch.mean(tp, -1) / ks  # [n_ks]
    rec = torch.mean(tp / n_gt_pos, -1)  # [n_ks]
    return pre, rec


def auc_compute(pred_loss: torch.Tensor, gt_label: torch.Tensor) -> torch.Tensor:
    n_users = pred_loss.size(0)
    auc = torch.empty(n_users)
    for u in range(n_users):
        pos_idx = torch.where(gt_label[u] != 0)[0]
        neg_idx = torch.where(gt_label[u] == 0)[0]
        pos_loss = pred_loss[u, pos_idx].reshape(-1, 1)
        neg_loss = pred_loss[u, neg_idx].reshape(1, -1)
        pred_diff = neg_loss - pos_loss
        auc[u] = torch.mean(torch.gt(pred_diff, 0).float()).cpu()
    return torch.mean(auc)


def _save_weight_internal(weight_dir: str, alg: str, time: str, params: tuple, epoch: int, max_keep_weights: int = 10,
                          is_best: bool = False):
    weight_dir = os.path.join(weight_dir, f'{alg}_{time}')
    target_path = os.path.join(weight_dir, 'latest.pth')
    utils.save_model(target_path, params, epoch, max_keep_weights=max_keep_weights,
                     ignore_file_regex='latest.pth')
    if is_best:
        shutil.copy(target_path, os.path.join(weight_dir, f'epoch_{epoch}.pth'))


_save_weight_func = None  # type: Optional[callable]


def save_weights(weights: tuple, epoch: int, max_keep_weights: int = 10, is_best: bool = False):
    if _save_weight_func is None:
        raise RuntimeError('call parse_common_args() before save_weights()')
    try:
        _save_weight_func(params=weights, epoch=epoch, max_keep_weights=max_keep_weights, is_best=is_best)
    except TypeError:
        raise RuntimeError('call parse_common_args() before save_weights()')


def parse_algorithm() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', choices=['dmf', 'dgmf', 'pdmf'], default='dmf')
    return parser.parse_known_args()[0].algorithm


def parse_common_args(extra_args_callback: Optional[List[Callable[[argparse.ArgumentParser], None]]]) \
        -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', choices=['dmf', 'dgmf', 'pdmf'], default='dmf')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', choices=['gowalla', 'foursquare'], default='gowalla')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--l2reg', type=float, default=0.1)
    parser.add_argument('--stddev', type=float, default=0.1)
    parser.add_argument('--yield_data', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--train_test_ratio', type=float, default=0.9)
    parser.add_argument('--test_sample_size', type=int, default=200)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--neighbor_selector', choices=['geo', 'user_sim', 'none'], default='geo')
    parser.add_argument('--sim_threshold', type=float, default=0.1)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--reset_log_dir', default=False, action='store_true')
    parser.add_argument('--weight_dir', type=str, default='weight')
    parser.add_argument('--save_weight_per_epoch', type=int, default=100)
    parser.add_argument('--max_keep_weights', type=int, default=5)
    parser.add_argument('--load_weight', type=str, default=None)
    parser.add_argument('--ensure_same_config', default=False, action='store_true')
    if extra_args_callback:
        for callback in extra_args_callback:
            callback(parser)
    # hook some args here
    args = parser.parse_args()
    utils.set_target_gpu(args.gpu)
    utils.fix_seed(args.seed)
    utils.configure_logging(args.verbose)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.weight_dir, exist_ok=True)
    os.makedirs(f'{args.log_dir}/{args.algorithm}', exist_ok=True)
    if args.reset_log_dir:
        utils.reset_dir(args.log_dir)
    global summary_writer
    time_formatter = '%y%m%d_%H%M%S'
    run_time = datetime.now().strftime(time_formatter)
    run_params = args.__dict__.copy()
    suppress_args = ['verbose', 'reset_log_dir', 'gpu', 'log_dir', 'weight_dir', 'save_weight_per_epoch',
                     'max_keep_weights', 'load_weight', 'epochs', 'ensure_same_config']
    if args.neighbor_selector == 'geo':
        run_params.pop('sim_threshold', None)
    for arg_name in suppress_args:
        run_params.pop(arg_name, None)
    run_id = ','.join(map(lambda kv: f'{kv[0]}={kv[1]}', run_params.items()))
    if args.load_weight is not None:
        if not args.load_weight.startswith(args.weight_dir):
            args.load_weight = os.path.join(args.weight_dir, args.load_weight)
        assert os.path.exists(args.load_weight), f'Path "{args.load_weight}" not exist'
        try:
            # fetch datetime from previous run
            prev_run_time = args.load_weight
            if os.path.isfile(prev_run_time):
                prev_run_time = os.path.dirname(prev_run_time)
            prev_run_time = os.path.basename(prev_run_time).split('_')
            prev_run_time = f'{prev_run_time[-2]}_{prev_run_time[-1]}'
            datetime.strptime(prev_run_time, time_formatter)
            # restore the time
            run_time = prev_run_time
            if args.ensure_same_config:
                assert os.path.isdir(f'{args.log_dir}/{args.algorithm}/{run_time},{run_id}'), \
                    'weight loaded with different config!'
        except ValueError:
            pass
    # summary will be merged if run settings are the same
    log_path = f'{args.log_dir}/{args.algorithm}/{run_time},{run_id}'
    if 'win' in sys.platform:
        # windows will raise OSError (NoSuchFile) when PATH exceeds MAX_PATH_LENGTH (~250 chars), UNC path is used here
        log_path = '\\\\?\\' + os.path.realpath(log_path)
    summary_writer = tb.SummaryWriter(log_path)
    global _save_weight_func
    _save_weight_func = partial(_save_weight_internal, weight_dir=args.weight_dir, alg=args.algorithm, time=run_time)
    return args
