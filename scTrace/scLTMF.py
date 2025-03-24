# Adapted from https://github.com/luliu8/Probabilistic-Matrix-Factorization-for-Music-Recommendation

import numpy as np
from scipy import stats
import time

class scLTMF(object):
    '''
    Kernelized Probabilistic Matrix Factorization as introduced in [Zhou12]
	Simplify to Probabilistic Matrix Factorization when kernel matrices are both diagonal
    '''
    def __init__(self, learning_rate=.005, regularization=0.02, n_epochs=20,
                 n_factors=100, min_flow=0, max_flow=1):
        self.lr = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.min_flow = min_flow
        self.max_flow = max_flow
        self.list_val_rmse = []
        self.list_train_rmse = []
        self.list_val_recall = []
        self.list_train_corr = []
        self.list_val_corr = []
        self.list_train_recall = []
        self.n_post_cells = 0
        self.n_pre_cells = 0
        self.n_flows = 0


    def _create_indicator_matrix(self):
        '''
        create the indicator matrix from training data
        where I[pre_cell,post_cell] indicate whether there is a flow from pre_cell to post_cell
        '''
        
        self.I = np.zeros((self.n_pre_cells, self.n_post_cells))
        for u,i in self.train[:,:2]:
            self.I[int(u), int(i)] = 1
        self.n_pre_flows = np.sum(self.I, axis=1)
        self.n_post_flows = np.sum(self.I, axis=0)


    def _sgd_initialization(self):
        """
        Initializes biases and latent factor matrixes.
        self.p (numpy array): pre_cells latent factor matrix.
        self.q (numpy array): post_cells latent factor matrix.
        """
        print("initalizing parameters for scTrace")
        self.p = np.random.normal(0, .1, (self.n_pre_cells, self.n_factors))
        self.q = np.random.normal(0, .1, (self.n_post_cells, self.n_factors))


    def _sgd_epoch_begin(self, epoch_ix):
        """
        Displays epoch starting log and returns its starting time.
        Args: epoch_ix: integer, epoch index.
        Returns: start (float): starting time of the current epoch.
        """
        start = time.time()
        end = '  | ' if epoch_ix < 9 else ' | '
        print('Epoch {}/{}'.format(epoch_ix+1, self.n_epochs), end=end)

        return start


    def _sgd_epoch_end(self, start, indicators):
        """
        Displays epoch ending log. If self.verbose compute and display validation metrics (loss/rmse/mae).
        Args1: start (float): starting time of the current epoch.
        Args2: indicators： floats, training rmse, validation rmse, training recall, validation recall.
        """
        end = time.time()
        train_rmse, val_rmse, train_recall, val_recall, train_corr, val_corr = indicators
        print('train_rmse: {:.3f}'.format(train_rmse), end=' - ')
        print('val_rmse: {:.3f}'.format(val_rmse), end=' - ')
        print('train_recall: {:.3f}'.format(train_recall), end=' - ')
        print('val_recall: {:.3f}'.format(val_recall), end=' - ')
        print('train_corr: {:.3f}'.format(train_corr), end=' - ')
        print('val_corr: {:.3f}'.format(val_corr), end=' - ')
        print('took {:.1f} sec'.format(end - start))


    def _compute_metrics(self, X, thres_consistency=0.3):
        """
        Computes rmse with current model parameters.
        Args: X (numpy array)
        """
        residuals = []
        y_pred = []
        non_missing, num_consistent, threshold = 0, 0, thres_consistency
        for i in range(X.shape[0]):
            pre_cell, post_cell, flow = int(X[i, 0]), int(X[i, 1]), X[i, 2]
            # predict global mean if pre_cell or post_cell is new
            if flow > threshold:
                non_missing += 1
            # if (pre_cell > -1) and (post_cell > -1):
            if pre_cell in self.pre_dict or post_cell in self.post_dict:
                pred = np.dot(self.p[pre_cell], self.q[post_cell])
            else:
                try:
                    n_neighbors = 10
                    neighbors_u = self.Su[pre_cell,].argsort()[::-1][1:n_neighbors + 1]
                    neighbors_i = self.Sv[post_cell,].argsort()[::-1][1:n_neighbors + 1]
                    all_sim, count = 0, 0
                    for u in neighbors_u:
                        for v in neighbors_i:
                            if u in self.pre_dict and v in self.post_dict:
                                all_sim += np.dot(self.p[u], self.q[v])
                                count += 1
                    pred = all_sim / count
                except ZeroDivisionError as e:
                    pred = np.dot(self.p[pre_cell], self.q[post_cell])
                    # print(e)
            residuals.append(flow - pred)
            y_pred.append(pred)
            if flow > threshold and pred > threshold:
                num_consistent += 1

        recall = num_consistent / non_missing
        residuals = np.array(residuals)
        loss = np.square(residuals).mean()
        rmse = np.sqrt(loss)
        y_true = X[:, 2]
        # pred_mat = np.dot(self.p, self.q.T)
        # y_pred = np.array([pred_mat[int(X[i, 0]), int(X[i, 1])] for i in range(X.shape[0])])
        # corr = np.corrcoef(y_pred, y_true)[0, 1]
        corr, p_value = stats.pearsonr(y_pred, y_true)

        return rmse, recall, corr


    def _sgd(self):
        """
        Performs SGD algorithm on training data, learns model parameters.
        Record all validation error and train error in a list.
        """
        self._sgd_initialization()
        self.min_val = 999
        # Run SGD
        for epoch_ix in range(self.n_epochs):
            start_time = self._sgd_epoch_begin(epoch_ix)

            if self.shuffle:
                np.random.shuffle(self.train)

            for i in range(self.n_flows):
                # 			pre_cell, post_cell, flow = self.train[i,:3].astype(np.int32)
                pre_cell, post_cell = self.train[i,:2].astype(np.int32)
                flow = self.train[i, 2]

                # Predict current flow
                pred = np.dot(self.p[pre_cell], self.q[post_cell])
                err = flow - pred
                # Update latent factors
                p_current = self.p[pre_cell]
                diff_p = err * self.q[post_cell] - (self.reg / self.n_pre_flows[pre_cell] / 2) * (self.Su[pre_cell] @ self.p + self.p[pre_cell])
                diff_q = err * p_current - (self.reg / self.n_post_flows[post_cell] / 2) * (self.Sv[post_cell] @ self.q + self.q[post_cell])
                self.p[pre_cell] += self.lr * diff_p
                self.q[post_cell] += self.lr * diff_q

            val_rmse, val_recall, val_corr = self._compute_metrics(self.val, self.thres_consistency)
            train_rmse, train_recall, train_corr = self._compute_metrics(self.train, self.thres_consistency)

            self.list_val_recall.append(val_recall)
            self.list_train_recall.append(train_recall)
            self.list_val_corr.append(val_corr)
            self.list_train_corr.append(train_corr)
            self.list_val_rmse.append(val_rmse)
            self.list_train_rmse.append(train_rmse)
            self.min_val = min(val_rmse, self.min_val)

            self._sgd_epoch_end(start_time, indicators=[train_rmse, val_rmse, train_recall, val_recall, train_corr, val_corr])

            # if early stopping and validation rmse didn't reduce enough, then break
            if self.early_stopping and self.list_val_rmse[-1] - self.min_val > 0.01:
#             if self.early_stopping and self.list_train_rmse[-2] - self.list_train_rmse[-1] < 0.01:
                break


    def fit(self, train = None, val = None, early_stopping=False, shuffle=True, n_pre_cell = 0, n_post_cell = 0,
            pre_cell_side = None, pre_cell_side_Su = None, post_cell_side = None, post_cell_side_Sv = None):
        #Learns model parameters.always require validation data
        self.early_stopping = early_stopping
        self.shuffle = shuffle
        self.thres_consistency = min(np.max(train['flow']) / 2,
                                     np.mean(train['flow']) - 2 * np.std(train['flow']))

        print('Preprocessing data...')

        self.n_pre_cells = n_pre_cell
        self.n_post_cells = n_post_cell
        self.n_flows = train.shape[0]
        self.train = train.values

        u_ids = train['u_id'].unique().tolist()
        i_ids = train['i_id'].unique().tolist()
        self.pre_dict = dict(zip(u_ids, [i for i in range(len(u_ids))]))
        self.post_dict = dict(zip(i_ids, [i for i in range(len(i_ids))]))

        self.val = val.values

        self._create_indicator_matrix()

        self.global_mean = np.mean(self.train[:,2])
        self.Su = np.diag(np.ones(self.n_pre_cells))
        self.Sv = np.diag(np.ones(self.n_post_cells))

        if pre_cell_side:
            print('Preparing pre_cell side information')
            # inverse of kernel matrix
            self.Su = pre_cell_side_Su
        if post_cell_side:
            print('Preparing post_cell side information')
            # inverse of kernel matrix
            self.Sv = post_cell_side_Sv

        self._sgd()

        return self
