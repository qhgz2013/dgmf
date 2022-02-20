import sys

import numpy as np
import torch
import alg_profile
import common
import neighbor
import utils
import scipy.sparse as sp
from typing import *
from functools import partial
from time import time
import os

_sp_cache = {}


def _proc_sp_matrix(sp_int_matrix_train: sp.spmatrix, sp_int_matrix_test: sp.spmatrix) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_key = f'sp_int_matrix_{id(sp_int_matrix_train)}_{id(sp_int_matrix_test)}'
    if cache_key in _sp_cache:
        return _sp_cache[cache_key]
    else:
        int_matrix_train = torch.tensor(np.asarray(sp_int_matrix_train.todense()), device='cuda')
        int_matrix_test = torch.tensor(np.asarray(sp_int_matrix_test.todense()), device='cuda')
        pos_idx_train = torch.where(int_matrix_train.view(-1) != 0)[0]
        pos_idx_test = torch.where(int_matrix_test.view(-1) != 0)[0]
        int_matrix = int_matrix_train
        int_matrix += int_matrix_test
        neg_idx = torch.where(int_matrix.view(-1) == 0)[0]
        data = pos_idx_train, pos_idx_test, neg_idx, int_matrix
        _sp_cache[cache_key] = data
        return data


def main():
    alg = common.parse_algorithm()
    profile = alg_profile.AlgorithmProfile.get_impl_profile(alg)
    args = common.parse_common_args(profile.argparse_callback)
    args_str = ', '.join(map(lambda x: f'{x[0]}={x[1]}', args.__dict__.items()))
    print(f'{alg} with args: {args_str}')
    if profile.profile_adjustment_by_args_callback is not None:
        profile.profile_adjustment_by_args_callback(profile, args)
    dataset = common.dataset_mapper[args.dataset]()
    train_mat, test_mat = common.split_train_test(dataset.sp_interaction_matrix, args.train_test_ratio)
    if args.neighbor_selector == 'geo':
        neighbor_selector = neighbor.GeoNeighborSelector(dataset.usr_city_dict, dataset.city_usr_sets)
    elif args.neighbor_selector == 'none':
        neighbor_selector = None
    else:
        neighbor_selector = neighbor.UserSimNeighborSelector(train_mat, args.sim_threshold)
    train_tensor = torch.tensor(np.asarray(train_mat.todense()), device='cuda')
    interaction_id = common.build_interaction_id(dataset.sp_interaction_matrix)
    test_item, test_label = common.fix_test_samples(test_mat, interaction_id, args.test_sample_size)
    test_item_tensor = torch.tensor(test_item, dtype=torch.int64, device='cuda')
    test_label_tensor = torch.tensor(test_label, device='cuda')
    if args.load_weight is None:
        params = profile.param_init_callback(args.k, dataset.n_users, dataset.n_pois, args.stddev)
        begin_epoch = 0
        max_p = max_r = max_auc = max_p_save = max_r_save = max_auc_save = 0
    else:
        begin_epoch, (params, max_p, max_r, max_auc) = utils.load_last_saved_model(args.load_weight,
                                                                                   ignore_file_regex='latest.pth')
        max_p_save, max_r_save, max_auc_save = max_p, max_r, max_auc
    pos_idx_train, pos_idx_test, neg_idx, int_matrix = _proc_sp_matrix(train_mat, test_mat)
    k = torch.tensor([5, 10], dtype=torch.int64, device='cuda')
    dataset_iterator = partial(profile.batch_data_generator, sp_int_matrix=train_mat, interaction_id=interaction_id,
                               neg_size=args.m, yield_data=args.yield_data, neg_weight=0.0)
    for epoch in range(begin_epoch + 1, args.epochs + 1):
        time_epoch_start = time()
        iter_func = partial(dataset_iterator, tqdm_kwargs={'leave': False, 'desc': f'{alg} epoch {epoch}', 'ncols': 80,
                                                           'mininterval': 0.2})
        for batch_data in common.batch_iterator(iter_func, batch_size=args.batch_size):
            batch_data_tensor = tuple([torch.tensor(x, device='cuda') for x in batch_data])
            profile.train_callback(params, batch_data_tensor, args.lr, args.l2reg, neighbor_selector, args.n)
        time_epoch_end = time()
        time_epoch_elapsed = time_epoch_end - time_epoch_start
        loss_pos_train, loss_pos_test, loss_neg = profile.epoch_loss_callback(params, pos_idx_train, pos_idx_test,
                                                                              neg_idx, int_matrix, torch.sum)
        loss_reg = common.l2_loss(args.l2reg, *params)
        print(f'{alg} epoch {epoch}: loss_pos_train: {loss_pos_train}, loss_pos_test: {loss_pos_test}, '
              f'loss_neg: {loss_neg}, loss_reg: {loss_reg}, Time: {time_epoch_elapsed}')
        metrics = profile.metric_pr_callback(params, k, train_tensor, test_item_tensor, test_label_tensor)
        pre_train, rec_train, pre_test, rec_test = [x.cpu().numpy() for x in metrics]
        auc = profile.metric_auc_callback(params, test_item_tensor, test_label_tensor)
        max_p, max_r, max_auc = np.maximum(max_p, pre_test), np.maximum(max_r, rec_test), np.maximum(auc, max_auc)
        print(f'Train: Pre: {pre_train}, Rec: {rec_train}; Test: Pre: {pre_test}, Rec: {rec_test}, AUC: {auc}')
        common.summary_writer.add_scalar('loss/pos/train', loss_pos_train, epoch)
        common.summary_writer.add_scalar('loss/pos/test', loss_pos_test, epoch)
        common.summary_writer.add_scalar('loss/reg', loss_reg, epoch)
        common.summary_writer.add_scalar('loss/neg', loss_neg, epoch)
        common.summary_writer.add_scalar('train_time', time_epoch_elapsed, epoch)
        common.summary_writer.add_scalar('metric/test/auc', auc, epoch)
        for i, k_ in enumerate(k.cpu().numpy()):
            common.summary_writer.add_scalar(f'metric/train/pre{k_}', pre_train[i], epoch)
            common.summary_writer.add_scalar(f'metric/test/pre{k_}', pre_test[i], epoch)
            common.summary_writer.add_scalar(f'metric/train/rec{k_}', rec_train[i], epoch)
            common.summary_writer.add_scalar(f'metric/test/rec{k_}', rec_test[i], epoch)
        if args.save_weight_per_epoch > 0 and epoch % args.save_weight_per_epoch == 0:
            is_best = max_auc > max_auc_save or any(max_p > max_p_save) or any(max_r > max_r_save)
            common.save_weights((params, max_p, max_r, max_auc), epoch, args.max_keep_weights, is_best)
            if is_best:
                max_p_save, max_r_save, max_auc_save = max_p, max_r, max_auc
    print(f'Finished running {alg} at epoch {args.epochs} with arg: {args_str}')
    train_time = os.path.basename(common.summary_writer.log_dir).split(',')[0]
    load_model_dir = os.path.join(args.weight_dir, f'{alg}_{train_time}', 'latest.pth')
    print(f'(Current args: {" ".join(sys.argv[1:])})')
    print(f'To resume this training, use --load_weight {load_model_dir}')


if __name__ == '__main__':
    main()
