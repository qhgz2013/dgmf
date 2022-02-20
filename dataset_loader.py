import numpy as np
import scipy.sparse as sp
from typing import *


class Dataset:
    def __init__(self, checkin_file, user_file):
        with open(checkin_file, 'r', encoding='utf8') as f:
            usrs = []
            pois = []
            cnts = []
            for line in f:
                if len(line) == 0:
                    continue
                u, poi, cnt = line.rstrip().split(' ')
                usrs.append(int(u))
                pois.append(int(poi))
                cnts.append(int(cnt))
            usrs = np.array(usrs, dtype=np.int32)
            pois = np.array(pois, dtype=np.int32)
            cnts = np.array(cnts, dtype=np.int32)
            self.sp_interaction_matrix = sp.coo_matrix((cnts, (usrs, pois)))
            self.n_users, self.n_pois = self.sp_interaction_matrix.shape
            self.n_interactions = self.sp_interaction_matrix.nnz
            self.interaction_matrix = np.asarray(self.sp_interaction_matrix.todense())
        with open(user_file, 'r', encoding='utf8') as f:
            usr_city_dict = {}
            city_usr_sets = {}
            for line in f:
                aggr_group = line.rstrip().split('\t')
                city = aggr_group[0]
                usrs = aggr_group[1:]
                usrs = [int(x) for x in usrs]
                for usr in usrs:
                    usr_city_dict[usr] = city
                city_usr_sets[city] = usrs
            self.usr_city_dict = usr_city_dict
            self.city_usr_sets = city_usr_sets


class GowallaDataset(Dataset):
    def __init__(self):
        checkin_file = 'data/gowalla_checkin.txt'
        user_file = 'data/gowalla_user_city_aggr.txt'
        super(GowallaDataset, self).__init__(checkin_file, user_file)


class FoursquareDataset(Dataset):
    def __init__(self):
        checkin_file = 'data/foursquare_checkin.txt'
        user_file = 'data/foursquare_user_city_aggr.txt'
        super(FoursquareDataset, self).__init__(checkin_file, user_file)
