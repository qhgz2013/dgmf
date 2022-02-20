from typing import *
import numpy as np
import scipy.sparse as sp


def _select_from_candidate(candidates: Union[np.ndarray, List[np.ndarray]], max_neighbors: int,
                           weight: Optional[Union[np.ndarray, List[np.ndarray]]] = None) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if not isinstance(candidates, list):
        # original selection
        if max_neighbors <= 0:
            max_neighbors = candidates.size
        if candidates.size <= max_neighbors:
            sampled = np.full(max_neighbors, -1, dtype=np.int32)  # -1 used as EMPTY
            sampled[:candidates.size] = candidates
            if weight is not None:
                sampled_weight = np.zeros_like(sampled)
                sampled_weight[:candidates.size] = weight
                return sampled, sampled_weight
            return sampled
        if weight is None:
            return np.random.choice(candidates, max_neighbors, replace=False)
        idx = np.random.choice(candidates.size, max_neighbors, replace=False)
        return candidates[idx], weight[idx]
    # list (batch) implementation
    assert len(candidates) > 0, 'empty candidates list!'
    if max_neighbors <= 0:
        max_neighbors = candidates[0].size
    sampled = np.full((len(candidates), max_neighbors), -1, dtype=np.int32)
    sampled_weight = np.zeros_like(sampled)
    if weight is None:
        for i, cur_candidates in enumerate(candidates):
            if cur_candidates.size <= max_neighbors:
                sampled[i, :cur_candidates.size] = cur_candidates
            else:
                sampled[i, :] = np.random.choice(cur_candidates, max_neighbors, replace=False)
        return sampled
    assert isinstance(weight, list) and len(weight) == len(candidates), 'invalid weight param'
    for i, (cur_candidates, cur_weight) in enumerate(zip(candidates, weight)):
        if cur_candidates.size < max_neighbors:
            sampled[i, :cur_candidates.size] = cur_candidates
            sampled_weight[i, :cur_candidates.size] = cur_weight
        else:
            idx = np.random.choice(cur_candidates.size, max_neighbors, replace=False)
            sampled[i, :] = cur_candidates[idx]
            sampled_weight[i, :] = cur_weight[idx]
    return sampled, sampled_weight


class NeighborSelector:
    def select(self, user: Union[int, np.ndarray], max_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class GeoNeighborSelector(NeighborSelector):
    def __init__(self, usr_city_dict: Dict[int, Any], city_usr_set: Dict[Any, List[int]]):
        candidates = {}
        for uid, city in usr_city_dict.items():
            user_list = city_usr_set[city].copy()
            user_list.remove(uid)  # remove self
            candidates[uid] = np.array(user_list, dtype=np.int32)
        self.candidates = candidates

    def select(self, user: Union[int, np.ndarray], max_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(user, int):
            ret_val = _select_from_candidate(self.candidates[user], max_neighbors=max_neighbors)
        else:
            ret_val = _select_from_candidate([self.candidates[u] for u in user], max_neighbors=max_neighbors)
        weight = np.greater_equal(ret_val, 0).astype(np.float32)
        return ret_val, weight


class UserSimNeighborSelector(NeighborSelector):
    def __init__(self, sp_int_matrix: sp.csr_matrix, filter_threshold: float = 0.2):
        item_sets = []
        n_users = sp_int_matrix.shape[0]
        col, row, data = [], [], []
        for u in range(n_users):
            item_u = set(sp_int_matrix.indices[sp_int_matrix.indptr[u]:sp_int_matrix.indptr[u+1]])
            item_sets.append(item_u)
        for u1 in range(n_users):
            item_u1 = item_sets[u1]
            for u2 in range(u1+1, n_users):
                item_u2 = item_sets[u2]
                val = len(item_u1.intersection(item_u2)) / np.sqrt(len(item_u1) * len(item_u2))
                if val >= filter_threshold:
                    col.extend([u1, u2])
                    row.extend([u2, u1])
                    data.extend([val, val])
        sp_us_mat = sp.csr_matrix((data, (row, col)), shape=(n_users, n_users), dtype=np.float32)
        self.sp_us_mat = sp_us_mat

    def select(self, user: Union[int, np.ndarray], max_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(user, int):
            candidates = self.sp_us_mat.indices[self.sp_us_mat.indptr[user]:self.sp_us_mat.indptr[user+1]]
            weights = self.sp_us_mat.data[self.sp_us_mat.indptr[user]:self.sp_us_mat.indptr[user+1]]
        else:
            candidates = [self.sp_us_mat.indices[self.sp_us_mat.indptr[x]:self.sp_us_mat.indptr[x+1]] for x in user]
            weights = [self.sp_us_mat.data[self.sp_us_mat.indptr[x]:self.sp_us_mat.indptr[x+1]] for x in user]
        return _select_from_candidate(candidates, max_neighbors=max_neighbors, weight=weights)


def main():
    # path tests
    c = np.array([1, 5, 3, 2, 8, 4])
    w = np.arange(1, 7)
    print(_select_from_candidate(c, 4))
    print(_select_from_candidate(c, 0))
    print(_select_from_candidate(c, 10))
    print(_select_from_candidate([c], 4))
    print(_select_from_candidate([c, c], 0))
    print(_select_from_candidate([c, c, c], 10))
    print(_select_from_candidate(c, 4, weight=w))
    print(_select_from_candidate(c, 0, weight=w))
    print(_select_from_candidate(c, 10, weight=w))
    print(_select_from_candidate([c, c], 4, weight=w))
    print(_select_from_candidate([c], 0, weight=w))
    print(_select_from_candidate([c, c, c], 11, weight=w))


if __name__ == '__main__':
    main()
