from typing import *
import argparse
import torch
import common
import neighbor
from functools import partial
import scipy.sparse as sp

tensor = torch.Tensor
loss_predict_type = Callable[[tuple, Optional[tensor]], tensor]
dmf_pdmf_param_type = Union[Tuple[tensor, tensor], Tuple[tensor, tensor, tensor]]


def argparse_neighbor_neg_sampling(parser: argparse.ArgumentParser):
    parser.add_argument('--m', type=int, default=3)  # neg sample size
    parser.add_argument('--n', type=int, default=5)  # number of users to be communicated


def weight_initialization_dgmf(k: int, n_users: int, n_pois: int, stddev: float = 0.1):
    w = torch.randn((n_users, k), device='cuda') * stddev
    p = torch.randn((n_users, k, n_pois), device='cuda') * stddev
    # noinspection PyTypeChecker
    q = torch.randn_like(p, device='cuda') * stddev
    return w, p, q


def loss_epoch(loss_pred_func: loss_predict_type, params: tuple, pos_idx_train: tensor, pos_idx_test: tensor,
               neg_idx: tensor, int_matrix: tensor, reduction: Callable[[Any], Any] = torch.sum) \
        -> Tuple[tensor, tensor, tensor]:
    loss = loss_pred_func(params, int_matrix).view(-1)
    return reduction(loss[pos_idx_train]), reduction(loss[pos_idx_test]), reduction(loss[neg_idx])


def metric_pr(loss_pred_func: loss_predict_type, params: tuple, k: tensor, train_int_matrix: tensor, test_item: tensor,
              test_label: tensor) -> Tuple[tensor, tensor, tensor, tensor]:
    loss = loss_pred_func(params, None)
    n_users = loss.size(0)
    usr_idx = torch.unsqueeze(torch.arange(n_users, dtype=torch.int64, device=loss.device), -1)
    train_ranking = torch.argsort(loss, -1)
    train_gt_label = train_int_matrix[usr_idx, train_ranking]
    pre_train, rec_train = common.pr_compute(train_gt_label, k)
    loss_test = loss[usr_idx, test_item]
    test_ranking = torch.argsort(loss_test, -1)
    test_gt_label = test_label[usr_idx, test_ranking]
    pre_test, rec_test = common.pr_compute(test_gt_label, k)
    return pre_train, rec_train, pre_test, rec_test


def metric_auc(loss_pred_func: loss_predict_type, params: tuple, test_item: tensor, test_label: tensor) -> tensor:
    loss = loss_pred_func(params, None)
    n_users = loss.size(0)
    loss = loss[torch.unsqueeze(torch.arange(n_users, dtype=torch.int64, device=loss.device), -1), test_item]
    return common.auc_compute(loss, test_label)


class AlgorithmProfile:
    _impl_dict = {}
    algorithm_name: Optional[str] = None
    # proto: argparse_callback(parser)
    argparse_callback: List[Callable[[argparse.ArgumentParser], Any]] = []
    profile_adjustment_by_args_callback: Optional[Callable[['AlgorithmProfile', argparse.Namespace], Any]] = None
    # proto: w, p, q = param_init_callback(k, n_users, n_pois)
    param_init_callback: Callable[[int, int, int, float], Any] = None
    # proto: train_callback((w, p, q), (batch_user, batch_poi, batch_y), lr, l2_reg, neighbor_selector, max_neighbors)
    train_callback: Callable[[tuple, tuple, float, float, Optional[neighbor.NeighborSelector], int], Any]
    # proto: loss = loss_prediction_callback((w, p, q), gt_dense_matrix)
    loss_prediction_callback: Callable[[tuple, Optional[tensor]], tensor]
    # proto: pos_loss_train, pos_loss_test, neg_loss = epoch_loss_callback((w, p, q), pos_idx_train, pos_idx_test,
    # neg_idx, int_matrix, reduction=torch.sum)
    epoch_loss_callback: Callable[[tuple, tensor, tensor, tensor, tensor, Callable[[Any], Any]],
                                  Tuple[tensor, tensor, tensor]]
    # proto: pre_train, rec_train, pre_test, rec_test = metric_pr_callback((w, p, q), k, train_int_matrix, test_item,
    # test_label)
    metric_pr_callback: Callable[[tuple, tensor, tensor, tensor, tensor], Tuple[tensor, tensor, tensor, tensor]]
    # proto: auc_test = metric_auc_callback((w, p, q), test_item, test_label)
    metric_auc_callback: Callable[[tuple, tensor, tensor], tensor]
    # proto: users, pois, labels = batch_data_generator(train_mat, interaction_id, neg_size, yield_data, neg_weight,
    # tqdm_kwargs)
    batch_data_generator: Callable[[Union[sp.csr_matrix, sp.coo_matrix], Set[int], int, bool, float,
                                    Optional[Dict[str, Any]]], Iterator[tuple]]

    def __init_subclass__(cls, **kwargs):
        cls._impl_dict[kwargs['algorithm_name']] = cls
        cls.algorithm_name = kwargs['algorithm_name']

    @classmethod
    def get_impl_profile(cls, alg_name: str) -> 'AlgorithmProfile':
        return cls._impl_dict[alg_name]


def compute_unique_grad(n_unique: int, inv_idx: tensor, grad: tensor) -> tensor:
    k = grad.size(1)
    unique_grad = torch.zeros((n_unique, k), dtype=grad.dtype, device=grad.device)
    unique_grad.scatter_add_(0, inv_idx.view(-1, 1).expand(-1, k), grad)
    return unique_grad


def update_neighbor(neighbor_selector: neighbor.NeighborSelector, max_neighbors: int, batch_user: tensor,
                    batch_pois: List[tensor], grads: List[tensor], p: tensor, lr: float):
    # batch_user, batch_poi: [batch_size], grad_pij: [batch_size, k], p: [n_users, k, n_pois]
    sampled_user, weight = neighbor_selector.select(batch_user.cpu().numpy(), max_neighbors)
    sampled_user = torch.tensor(sampled_user, device='cuda', dtype=torch.int64)  # [batch_size, max_neighbors]
    weight = torch.tensor(weight, device='cuda', dtype=torch.float)  # [batch_size, max_neighbors]
    p = torch.cat([p, torch.zeros((1, p.size(1), p.size(2)), dtype=p.dtype, device=p.device)])  # p[-1] not used
    # TEST
    for batch_poi, grad in zip(batch_pois, grads):
        batch_poi_reshape = torch.reshape(batch_poi, (-1, 1))
        grad_neighbor = torch.unsqueeze(grad, 1)  # [batch_size, 1, k] -> [batch_size, max_neighbors, k]
        # # original impl
        # a2 = torch.empty_like(a1)
        # g2 = torch.empty_like(g1)
        # for i, u in enumerate(batch_user.cpu().numpy()):
        #     user, weights = sampled_user[i], weight[i]  # [n_neighbors]
        #     weights = torch.unsqueeze(torch.tensor(weights, device='cuda'), -1)  # [n_neighbors, 1]
        #     grad_pij_neighbor = torch.unsqueeze(grad[i], 0)  # grad_pij[i]: [k] -> [1, k]
        #     # p[user, :, batch_poi[i]] -= lr * weights * grad_pij_neighbor  # [n_neighbors, k]
        #     a2[i] = p[user, :, batch_poi[i]]
        #     g2[i] = lr * weights * grad_pij_neighbor
        # lparam: [batch_size, max_neighbors, k]
        p[sampled_user, :, batch_poi_reshape] -= lr * torch.unsqueeze(weight, -1) * grad_neighbor


def loss_predict_dgmf(params: Tuple[tensor, tensor, tensor], gt_matrix: Optional[tensor] = None) -> tensor:
    w, p, q = params
    v = p + q
    pred = torch.sum(torch.unsqueeze(w, -1) * v, 1)  # [n_users, n_pois]
    if gt_matrix is None:
        return -pred
    prob = torch.sigmoid(pred)
    gt_matrix = torch.gt(gt_matrix, 0).float()
    # noinspection PyTypeChecker
    return -gt_matrix * torch.log(prob + 1e-10) - (1 - gt_matrix) * torch.log(1 - prob + 1e-10)


def train_dgmf_sgd(params: Tuple[tensor, tensor, tensor], batch_data: Tuple[tensor, tensor, tensor, tensor],
                   lr: float, l2_reg: float, neighbor_selector: Optional[neighbor.NeighborSelector] = None,
                   max_neighbors: int = 5, grad_clip_value: Optional[float] = 1.0):
    w, p, q = params
    batch_user, batch_poi_i, batch_poi_j, _ = batch_data
    batch_user, batch_poi_i = batch_user.to(torch.int64), batch_poi_i.to(torch.int64)
    batch_poi_j = batch_poi_j.to(torch.int64)
    batch_size = batch_user.size(0)

    # batch data
    batch_wi = w[batch_user, :]  # [batch_size, k]
    batch_pij1 = p[batch_user, :, batch_poi_i]  # [batch_size, k]
    batch_pij2 = p[batch_user, :, batch_poi_j]
    batch_qij1 = q[batch_user, :, batch_poi_i]
    batch_qij2 = q[batch_user, :, batch_poi_j]
    vij1 = batch_pij1 + batch_qij1
    vij2 = batch_pij2 + batch_qij2
    rij1_pred = torch.sum(batch_wi * vij1, -1)  # [batch_size]
    rij2_pred = torch.sum(batch_wi * vij2, -1)
    batch_rij_diff = rij1_pred - rij2_pred

    # compute gradients
    # noinspection PyTypeChecker
    term1 = torch.unsqueeze(-1 / (1 + torch.exp(batch_rij_diff)), -1)  # -1 / (1 + e^(r_ui - r_uj))
    grad_wi = term1 * (vij1 - vij2) + l2_reg * batch_wi
    term2 = term1 * batch_wi
    grad_pij1 = term2 + l2_reg * batch_pij1
    grad_pij2 = -term2 + l2_reg * batch_pij2
    grad_qij1 = term2 + l2_reg * batch_qij1
    grad_qij2 = -term2 + l2_reg * batch_qij2

    # merge update with grad clipping
    unique_user_w, inv_idx_w = torch.unique_consecutive(batch_user, return_inverse=True)
    unique_grad_w = compute_unique_grad(unique_user_w.size(0), inv_idx_w, grad_wi)
    unique_i, inv_idx_i = torch.unique(torch.stack([batch_user, batch_poi_i]), dim=1, return_inverse=True)
    unique_j, inv_idx_j = torch.unique(torch.stack([batch_user, batch_poi_j]), dim=1, return_inverse=True)
    # noinspection DuplicatedCode
    if unique_i.size(1) == batch_size:
        unique_grad_pij1, unique_grad_qij1, unique_user_i, unique_poi_i = grad_pij1, grad_pij2, batch_user, batch_poi_i
    else:
        unique_grad_pij1 = compute_unique_grad(unique_i.size(1), inv_idx_i, grad_pij1)
        unique_grad_qij1 = compute_unique_grad(unique_i.size(1), inv_idx_i, grad_qij1)
        unique_user_i, unique_poi_i = unique_i[0], unique_i[1]
    # noinspection DuplicatedCode
    if unique_j.size(1) == batch_size:
        unique_grad_pij2, unique_grad_qij2, unique_user_j, unique_poi_j = grad_pij2, grad_qij2, batch_user, batch_poi_j
    else:
        unique_grad_pij2 = compute_unique_grad(unique_j.size(1), inv_idx_j, grad_pij2)
        unique_grad_qij2 = compute_unique_grad(unique_j.size(1), inv_idx_j, grad_qij2)
        unique_user_j, unique_poi_j = unique_j[0], unique_j[1]
    if grad_clip_value is not None:
        grads = [unique_grad_w, unique_grad_pij1, unique_grad_pij2, unique_grad_qij1, unique_grad_qij2]
        map(lambda x: torch.clip_(x, -grad_clip_value, grad_clip_value), grads)

    # apply SGD
    w[unique_user_w, :] -= lr * unique_grad_w
    p[unique_user_i, :, unique_poi_i] -= lr * unique_grad_pij1
    p[unique_user_j, :, unique_poi_j] -= lr * unique_grad_pij2
    q[unique_user_i, :, unique_poi_i] -= lr * unique_grad_qij1
    q[unique_user_j, :, unique_poi_j] -= lr * unique_grad_qij2
    if neighbor_selector is None:
        return
    update_neighbor(neighbor_selector, max_neighbors, batch_user, [batch_poi_i, batch_poi_j],
                    [grad_pij1, grad_pij2], p, lr)


def argparse_dgmf(parser: argparse.ArgumentParser):
    parser.add_argument('--pos_pair', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=None)


def profile_adjust_dgmf(profile: AlgorithmProfile, args: argparse.Namespace) -> NoReturn:
    profile.batch_data_generator = partial(profile.batch_data_generator, max_pair_per_interaction=args.pos_pair)
    profile.train_callback = partial(profile.train_callback, grad_clip_value=args.grad_clip)


class DGMFProfile(AlgorithmProfile, algorithm_name='dgmf'):
    argparse_callback = [argparse_neighbor_neg_sampling, argparse_dgmf]
    profile_adjustment_by_args_callback = profile_adjust_dgmf
    param_init_callback = weight_initialization_dgmf
    train_callback = train_dgmf_sgd
    loss_prediction_callback = loss_predict_dgmf
    epoch_loss_callback = partial(loss_epoch, loss_prediction_callback)
    metric_pr_callback = partial(metric_pr, loss_prediction_callback)
    metric_auc_callback = partial(metric_auc, loss_prediction_callback)
    batch_data_generator = common.pairwise_interaction_iterator
