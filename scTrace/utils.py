import numpy as np
import pandas as pd
import scanpy as sc
import math
from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
import networkx as nx
from node2vec import Node2Vec
from scipy.stats import ranksums, mannwhitneyu
from tqdm import trange
from collections import Counter
import pickle


def preProcess(scobj, n_hvgs=2000, log_normalize=True):
    # sc.pp.filter_cells(scobj, min_genes=100)
    # sc.pp.filter_genes(scobj, min_cells=3)
    if log_normalize:
        sc.pp.normalize_total(scobj, target_sum=1e4)
        sc.pp.log1p(scobj)
    sc.pp.highly_variable_genes(scobj, n_top_genes=n_hvgs)
    return scobj


def runScanpy(scobj, n_neighbors=10, n_pcs=40, regress=True):
    scobj.raw = scobj
    scobj = scobj[:, scobj.var.highly_variable]
    if regress:
        sc.pp.regress_out(scobj, ['nCount_RNA', 'percent.mt'])
    sc.pp.scale(scobj, max_value=10)
    sc.tl.pca(scobj, svd_solver='arpack')
    sc.pp.neighbors(scobj, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(scobj)
    return scobj


def getSimilarityMatrix(data_pre, data_pos, method='Pearson'):
    data_comb = np.vstack((data_pre.raw.to_adata()[:, data_pre.var.index[data_pre.var.highly_variable]].X.toarray(),
                           data_pos.raw.to_adata()[:, data_pos.var.index[data_pos.var.highly_variable]].X.toarray()))
    n_pre = data_pre.shape[0]
    n_pos = data_pos.shape[0]

    all_sim = None
    if method == 'Pearson':
        all_sim = np.corrcoef(data_comb)
    elif method == 'Cosine':
        all_sim = 1 - pairwise.cosine_distances(data_comb)

    cross_sim = all_sim[0:n_pre, n_pre:(n_pre + n_pos)]
    # pre_sim = all_sim[0:n_pre, 0:n_pre]
    # pos_sim = all_sim[n_pre:(n_pre + n_pos), n_pre:(n_pre + n_pos)]

    # return data_comb, cross_sim, pre_sim, pos_sim
    return cross_sim


def get1DMat(mat, run_type):
    if run_type == 'cross':
        values = mat.reshape((1, -1))[0]
    else:
        values = mat[np.tri(mat.shape[0], mat.shape[1], k=-1, dtype=bool)]
        if len(values) == 0:
            values = [1]
    return (values)


def getLineageMatrix(bars, bars2=None, ignore_bars=None):
    bars = pd.Series(bars, dtype='category')
    # Cross-timepoint
    if bars2 is not None:
        bars2 = pd.Series(bars2, dtype='category')
        lin_mat = pd.DataFrame(np.zeros((len(bars), len(bars2))))
        clonotype_mat = pd.DataFrame(np.empty((len(bars), len(bars2)), dtype=str))
        # Intersection of 2 timepoints
        comm_bars = list(set(bars).intersection(set(bars2)) - {np.nan})
        if ignore_bars:
            comm_bars = list(set(comm_bars) - set(ignore_bars))
        # Search in common barcodes
        for bar in comm_bars:
            t_ix1 = np.where(bars == bar)[0]
            t_ix2 = np.where(bars2 == bar)[0]
            lin_mat.iloc[t_ix1, t_ix2] = 1
            clonotype_mat.iloc[t_ix1, t_ix2] = bar
    # Within-timepoint (symmetry)
    else:
        lin_mat = pd.DataFrame(np.zeros((len(bars), len(bars))))
        clonotype_mat = None
        # clonotype_mat = pd.DataFrame(np.empty((len(bars), len(bars)), dtype=object))
        comm_bars = set(bars) - {np.nan}
        if ignore_bars:
            comm_bars = comm_bars - set(ignore_bars)
        for bar in list(comm_bars):
            t_ix1 = np.where(bars == bar)[0]
            lin_mat.iloc[t_ix1, t_ix1] = 1
            # clonotype_mat.iloc[t_ix1, t_ix1] = bar
    # return (np.array(lin_mat))
    return np.array(lin_mat), np.array(clonotype_mat)


def getCrossLineageDensity(cross_lin_mat):
    n1, n2, n3, n4 = (cross_lin_mat.shape[0], sum(cross_lin_mat.sum(axis=1) != 0),
                      cross_lin_mat.shape[1], sum(cross_lin_mat.sum(axis=0) != 0))
    print("Number of cells in the former time point: ", n1)
    print("Number of cells with flow-out information: ", n2)
    print("Flow-out density: ", n2 / n1)
    print("Number of cells in the latter time point: ", n3)
    print("Number of cells with flow-in information: ", n4)
    print("Flow-in density: ", n4 / n3)
    return n2 / n1, n4 / n3


def plotSimilarityCompare(cross_sim, cross_lin_mat, title, savePath):
    plt.figure(figsize=(3.5, 2.5), dpi=300)
    # plt.figure(figsize=(5, 4), dpi=300)

    within_clone = np.array(coo_matrix(np.multiply(cross_sim, cross_lin_mat)).data)
    other_value = np.array(coo_matrix(np.multiply(cross_sim, 1 - cross_lin_mat)).data)
    within_clone = within_clone[within_clone <= 0.99]
    other_value = other_value[other_value <= 0.99]
    print(np.max(within_clone), np.max(other_value))
    plt.hist(other_value, color='#F9DF91', fill='#F9DF91', density=True, log=True,
             bins=20, alpha=0.6, label='Cross-clone')
    plt.hist(within_clone, color='#A8CFE8', fill='#A8CFE8', density=True, log=True,
             bins=20, alpha=0.6, label='Within-clone')

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Density (log-scale)', fontsize=12)
    plt.xlabel('Transcriptomic similarity', fontsize=12)
    plt.title(title, fontsize=14)
    # plt.legend()

    plt.tight_layout()
    plt.savefig(savePath)


def inv_node2vec_kernel(adj_matrix):
    print("computing node2vec kernel matrix")
    n_nodes = adj_matrix.shape[0]
    G = nx.from_numpy_array(adj_matrix)
    dimensions = 32
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=20, num_walks=100)
    # Embed nodes
    params = {'window': 10, 'min_count': 1, 'batch_words': 4}
    print("fitting node2vec model")
    model = node2vec.fit(**params)
    print("generating embedding matrix")
    embed_matrix = np.zeros((n_nodes, dimensions))
    for i in trange(n_nodes):
        if str(i) in model.wv:
            embed_matrix[i] = model.wv[str(i)]
    K_inv = np.linalg.inv(pairwise.rbf_kernel(embed_matrix))
    return K_inv


def train_valid_split(data, ratio, type="stratified"):
    print("Splitting train-validation set")
    sparse_mat_len = data.shape[0]
    np.random.seed(123)
    if type == "stratified":
        all_clonotypes = list(data['clonotype'])
        val_index = np.array([])
        for clonotype in np.unique(all_clonotypes):
            clone_index = np.array(data[data['clonotype'] == clonotype].index)
            new_index = np.random.choice(clone_index, int(ratio * len(clone_index)), replace=False)
            val_index = np.append(val_index, new_index)

    else:
        # type == "random"
        val_index = np.random.choice(np.arange(sparse_mat_len), int(ratio * sparse_mat_len), replace=False)

    val_index = np.array(val_index, dtype=int)
    val_index.sort()
    train_index = np.setdiff1d(np.arange(sparse_mat_len), val_index)
    train_df, val_df = data.loc[data.index[train_index],], data.loc[data.index[val_index],]
    return train_df, val_df


def grid_search(model, train_df, val_df, n_pre_cell, n_post_cell, n_factor, n_epoch, Su, Sv,
                bool_pre_side=True, bool_post_side=True, learning_rate=0.01, regularization=0.0001):
    params = {'regularization': 0.3, 'n_factors': n_factor, 'n_epochs': n_epoch}
    result = {'min_val': 999, 'lr': None, 'reg': None}
    # for lr in [0.1, 0.05, 0.01]:# , 0.005, 0.0001]:
    for lr in [learning_rate]:
        #    for reg in [1, 0.1, 0.01, 0.001, 0.0001, 0.00005]:
        for reg in [regularization]:
            params['learning_rate'] = lr
            params['regularization'] = reg
            m = model(**params)
            print(m.reg)
            print("trying learning rate : {}, regularization : {}".format(lr, reg))
            # fit with early stopping to obtain the best model
            m.fit(train=train_df, val=val_df, early_stopping=True, n_pre_cell=n_pre_cell, n_post_cell=n_post_cell,
                  pre_cell_side=bool_pre_side, pre_cell_side_Su=Su, post_cell_side=bool_post_side, post_cell_side_Sv=Sv)
            if m.min_val < result['min_val']:
                result['min_val'] = m.min_val
                result['lr'] = lr
                result['reg'] = reg

    print('best lr: {}, best reg: {}, min val rmse: {}'.format(result['lr'], result['reg'], result['min_val']))
    return result, m


def plot_metrics(model, savePath, run_label_time, showName):
    val_recall, train_recall = model.list_val_recall, model.list_train_recall
    val_rmse, train_rmse = model.list_val_rmse, model.list_train_rmse
    idx1, idx2 = np.argmin(val_rmse), np.argmax(val_recall)
    epochs = list(range(len(val_recall)))
    fig, axs = plt.subplots(1, 2, figsize=(9, 5))

    # RMSE-Epoch
    axs[0].plot(epochs, val_rmse, label='Validation', color='blue', marker='.')
    axs[0].plot(epochs, train_rmse, label='Training', color='orange', marker='.')
    # axs[0].set_title('RMSE: '+ run_label_time)
    axs[0].set_xlabel('Epoch', fontsize=18)
    axs[0].set_ylabel('RMSE', fontsize=18)
    axs[0].legend(loc='upper right', fontsize=18)
    axs[0].grid(False)

    #     axs[0].text(max(epochs) / 2, max(train_rmse) / 2, f"Min RMSE: {format(val_rmse[idx1], '.3f')}", ha='left',
    #                 bbox=dict(facecolor='white', edgecolor='black'))

    # Recall-Epoch
    axs[1].plot(epochs, val_recall, label='Validation', color='blue', marker='.')
    axs[1].plot(epochs, train_recall, label='Training', color='orange', marker='.')
    # axs[1].set_title('Recall: ' + run_label_time)
    axs[1].set_xlabel('Epoch', fontsize=18)
    axs[1].set_ylabel('Recall', fontsize=18)
    axs[1].legend(loc='lower right', fontsize=18)
    axs[1].grid(False)

    #     axs[1].text(max(epochs) / 2, max(train_recall) / 2, f"Max Recall: {format(val_recall[idx2], '.3f')}", ha='left',
    #                 bbox=dict(facecolor='white', edgecolor='black'))

    plt.suptitle("Training process of " + showName, fontsize=20)
    plt.tight_layout()
    # plt.show()
    plt.savefig(savePath + run_label_time + '_metrics.png', dpi=300, bbox_inches='tight')
    # plt.close()
    return val_rmse[idx1], val_recall[idx2]


def plotFittingResults(y_pred, y_true, y_pred_val, y_true_val, savePath, run_label_time, showName):
    # Need to add correlation coefficient in these figures

    plt.figure(figsize=(3, 3), dpi=300)
    plt.scatter(y_pred, y_true, alpha=0.1, s=5, color='#884b91')
    corr = np.corrcoef(y_pred, y_true)[0, 1]
    plt.plot([min(min(y_pred), min(y_true)), max(max(y_pred), max(y_true))],
             [min(min(y_pred), min(y_true)), max(max(y_pred), max(y_true))],
             linestyle='--', color='gray', linewidth=0.7)
    plt.text(0.95, 0.05, f'Correlation (Train): {corr:.3f}',
             fontsize=10, transform=plt.gca().transAxes,
             horizontalalignment='right',
             verticalalignment='bottom')
    plt.title(showName)
    plt.xlabel('Predicted values')
    plt.ylabel('Raw matrix values')
    plt.tight_layout()
    plt.savefig(savePath + run_label_time + '-RawMatrixValues_compare.png')

    plt.figure(figsize=(3, 3), dpi=300)
    plt.scatter(y_pred_val, y_true_val, alpha=0.1, s=5, color='#884b91')
    corr_val = np.corrcoef(y_pred_val, y_true_val)[0, 1]
    plt.plot([min(min(y_pred_val), min(y_true_val)), max(max(y_pred_val), max(y_true_val))],
             [min(min(y_pred_val), min(y_true_val)), max(max(y_pred_val), max(y_true_val))],
             linestyle='--', color='gray', linewidth=0.7)
    plt.text(0.95, 0.05, f'Correlation (Val): {corr_val:.3f}',
             fontsize=10, transform=plt.gca().transAxes,
             horizontalalignment='right',
             verticalalignment='bottom')
    plt.title(showName)
    plt.xlabel('Predicted values')
    plt.ylabel('Raw matrix values')
    plt.tight_layout()
    plt.savefig(savePath + run_label_time + '-RawMatrixValues_compare_val.png')

    return corr


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculateFractions(scd_obj):
    pre_fractions = np.array([sum(scd_obj.data_pre.obs['cluster'] == str(i)) for i in range(scd_obj.n_clus[0])])
    pre_fractions = pre_fractions / sum(pre_fractions)
    pos_fractions = np.array([sum(scd_obj.data_pos.obs['cluster'] == str(i)) for i in range(scd_obj.n_clus[1])])
    pos_fractions = pos_fractions / sum(pos_fractions)
    return pre_fractions, pos_fractions


def plotFlowSankey(flow_info, pre_colors, pos_colors, pre_fractions=None, pos_fractions=None,
                   label_position='mid', pre_label=None, pos_label=None, label_size=14, title=None,
                   show_cls=True, figwidth=3.5, figheight=4, saveFig=False, saveName='flow.png'):
    pre_names = flow_info['s'].unique()
    pre_names.sort()
    if pre_fractions is None:
        pre_fractions = np.array([flow_info['s_pm'][flow_info['s'] == item].sum() for item in pre_names])

    pos_names = flow_info['t'].unique()
    pos_names.sort()
    if pos_fractions is None:
        pos_fractions = np.array([flow_info['t_pm'][flow_info['t'] == item].sum() for item in pos_names])

    flow_info['s_ix'] = [np.where(pre_names == x)[0][0] for x in flow_info['s']]
    flow_info['t_ix'] = [np.where(pos_names == x)[0][0] for x in flow_info['t']]

    fig, ax = plt.subplots(figsize=(figwidth, figheight), tight_layout=True, facecolor='none')

    pre_x, pos_x = 0, 1
    if label_position == 'mid':
        pre_x, pos_x = 0, 1
    elif label_position == 'twoside':
        pre_x, pos_x = -0.15, 1.15

    for i in range(len(pre_fractions)):
        bottom = pre_fractions[(i + 1):].sum()
        #         bottom = 1 - pre_fractions[0:(i+1)].sum()
        rectangle = ax.bar(x=[0], height=pre_fractions[i], bottom=bottom, color=pre_colors[i],
                           edgecolor='black', fill=True, linewidth=0.7, width=0.16)
        text_y = rectangle[0].get_height() / 2 + bottom
        if show_cls:
            ax.text(x=pre_x, y=text_y, s=str(pre_names[i]), horizontalalignment='center', verticalalignment='center',
                    fontsize=label_size)
    for i in range(len(pos_fractions)):
        bottom = pos_fractions[(i + 1):].sum()
        #         bottom = 1 - pos_fractions[0:(i+1)].sum()
        rectangle = ax.bar(x=[1], height=pos_fractions[i], bottom=bottom, color=pos_colors[i],
                           edgecolor='black', fill=True, linewidth=0.7, width=0.16)
        text_y = rectangle[0].get_height() / 2 + bottom
        if show_cls:
            ax.text(x=pos_x, y=text_y, s=str(pos_names[i]), horizontalalignment='center', verticalalignment='center',
                    fontsize=label_size)

    if pre_label is not None:
        ax.text(x=0, y=-0.05, s=pre_label, horizontalalignment='center', verticalalignment='center', fontsize=14.5)
    if pos_label is not None:
        ax.text(x=1, y=-0.05, s=pos_label, horizontalalignment='center', verticalalignment='center', fontsize=14.5)

    xs = np.linspace(-5, 5, num=100)
    ys = np.array([sigmoid(x) for x in xs])
    xs = xs / 10 + 0.5
    xs *= 0.83
    xs += 0.085
    #     y_start_record = [1 - pre_fractions[0:ii].sum() for ii in range(len(pre_fractions))]
    #     y_end_record = [1 - pos_fractions[0:ii].sum() for ii in range(len(pos_fractions))]
    y_start_record = [pre_fractions[ii:].sum() for ii in range(len(pre_fractions))]
    y_end_record = [pos_fractions[ii:].sum() for ii in range(len(pos_fractions))]
    y_up_start, y_dw_start = 1, 1
    y_up_end, y_dw_end = 1, 1
    axi = 0
    for si in range(len(pre_fractions)):
        cur_flow_info = flow_info.loc[flow_info['s_ix'] == si, :]
        if cur_flow_info.shape[0] > 0:
            for fi in range(cur_flow_info.shape[0]):
                y_up_start = y_start_record[si]
                y_dw_start = y_up_start - cur_flow_info['s_pm'].iloc[fi]
                y_start_record[si] = y_dw_start

                ti = cur_flow_info['t_ix'].iloc[fi]
                y_up_end = y_end_record[ti]
                y_dw_end = y_up_end - cur_flow_info['t_pm'].iloc[fi]
                y_end_record[ti] = y_dw_end

                y_up_start -= 0.0005
                y_dw_start += 0.001
                y_up_end -= 0.0005
                y_dw_end += 0.001

                ys_up = y_up_start + (y_up_end - y_up_start) * ys
                ys_dw = y_dw_start + (y_dw_end - y_dw_start) * ys

                color_s_t = [pre_colors[si], pos_colors[ti]]
                cmap = LinearSegmentedColormap.from_list('mycmap', [color_s_t[0], color_s_t[1]])
                grad_colors = cmap(np.linspace(0, 1, len(xs) - 1))
                grad_colors = [rgb2hex(color) for color in grad_colors]
                for pi in range(len(xs) - 1):
                    ax.fill_between(xs[pi:(pi + 2)], ys_dw[pi:(pi + 2)], ys_up[pi:(pi + 2)], alpha=0.7,
                                    color=grad_colors[pi], edgecolor=None)

            if pre_fractions[(si + 1):].sum() < y_dw_start - 0.001:
                rectangle = ax.bar(x=[0], height=y_dw_start - 0.001 - pre_fractions[(si + 1):].sum() - 0.01,
                                   bottom=pre_fractions[(si + 1):].sum() + 0.005, color='lightgrey',
                                   edgecolor='grey', fill=True, hatch='//', alpha=0.6, linewidth=0.7, width=0.14)

        elif cur_flow_info.shape[0] == 0:
            y_up_start = y_start_record[si]
            y_dw_start = y_up_start - pre_fractions[si]
            y_start_record[si] = y_dw_start

            y_up_end = y_dw_end
            y_dw_end = y_up_end - 0

            y_up_start -= 0.0005
            y_dw_start += 0.001
            y_up_end -= 0.0005
            y_dw_end = y_up_end

            ys_up = y_up_start + (y_up_end - y_up_start) * ys
            ys_dw = y_dw_start + (y_dw_end - y_dw_start) * ys

            ax.fill_between(xs, ys_dw, ys_up, alpha=0.7,
                            color=pre_colors[si])

            color_s_t = [pre_colors[si], 'white']
            cmap = LinearSegmentedColormap.from_list('mycmap', [color_s_t[0], color_s_t[1]])
            grad_colors = cmap(np.linspace(0, 1, len(xs) - 1))
            grad_colors = [rgb2hex(color) for color in grad_colors]
            for pi in range(len(xs) - 1):
                ax.fill_between(xs[pi:(pi + 2)], ys_dw[pi:(pi + 2)], ys_up[pi:(pi + 2)], alpha=0.7,
                                color=grad_colors[pi], edgecolor=None)

    for ti in range(len(pos_fractions)):
        if pos_fractions[(ti + 1):].sum() < y_end_record[ti] - 0.001:
            rectangle = ax.bar(x=[1], height=y_end_record[ti] - 0.001 - pos_fractions[(ti + 1):].sum() - 0.01,
                               bottom=pos_fractions[(ti + 1):].sum() + 0.005, color='lightgrey',
                               edgecolor='grey', fill=True, hatch='//', alpha=0.6, linewidth=0.7, width=0.14)

    for pos in ['right', 'top', 'bottom', 'left']:
        ax.spines[pos].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_ylim(-0.01, 1.01)
    ax.patch.set_alpha(0)

    if title is not None:
        ax.set_title(title)

    if saveFig:
        fig.savefig(saveName, dpi=300, transparent=True, facecolor='none', edgecolor='none', pad_inches=0.0)

    return (fig)


def generateDEGs(cur_expr, index, cluster, fate_str, filter_signif = True):
    de_df = pd.DataFrame({'Gene': pd.DataFrame(cur_expr.uns['rank_genes_groups']['names']).iloc[:, index],
                        'Cluster': cluster,
                        'Fate': fate_str, # str(si) + '->' + str(ti) or 'FlowTo' + str(ti)
                        'scores': pd.DataFrame(cur_expr.uns['rank_genes_groups']['scores']).iloc[:, index],
                        'logfoldchanges': pd.DataFrame(cur_expr.uns['rank_genes_groups']['logfoldchanges']).iloc[:, index],
                        'pvals': pd.DataFrame(cur_expr.uns['rank_genes_groups']['pvals']).iloc[:, index],
                        'pvals_adj': pd.DataFrame(cur_expr.uns['rank_genes_groups']['pvals_adj']).iloc[:, index]})
    if filter_signif:
        de_df = de_df[(de_df['pvals_adj'] < 0.05) & (de_df['logfoldchanges'] > 0)]
    else:
        de_df = de_df[de_df['logfoldchanges'] > 0]
    return de_df


def getColorMap(items, special_case=None, default_palette=("#43D9FE", "#E78AC3", "#FEC643", "#A6D854",
                                                           "#FE6943", "#E5C494", "#33AEB1", "#FFEC1A",
                                                           "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3",
                                                           "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3",
                                                           "#D5BB67", "#6ACC64", "#D65F5F", "#82C6E2",
                                                           "#DC7EC0", "#4878D0", '#B5A2E0', '#F9B475',
                                                           '#50C7CA', '#CF747A', '#63AFF0', '#8792AF', '#E0CB00')):
    # default_palette = sns.color_palette("muted") + sns.color_palette("Set2")
    custom_palette = {}
    items = items.astype('category')
    for category in items.cat.categories:
        custom_palette[category] = default_palette[len(custom_palette) % len(default_palette)]
    if special_case is not None:
        custom_palette[special_case] = 'lightgray'

    return custom_palette


def plotCellFate(adata, savePath, run_label_time, cls_colname, fate_colname,
                 special_case="Missing", png_name="_cellfate-umap.png"):
    cls_colors = getColorMap(adata.obs[cls_colname])
    fate_colors = getColorMap(adata.obs[fate_colname], special_case)
    cur_colors = cls_colors.update(fate_colors) # cur_colors = cls_colors | fate_colors in Python3.9+
    with plt.rc_context({'figure.figsize': (3, 3)}):
        sc.pl.umap(adata, color=[cls_colname, fate_colname], palette=cur_colors, show=False)
    plt.savefig(savePath + run_label_time + png_name, dpi=300, bbox_inches='tight')
    plt.close()


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Suppose we need to remap 'old_module_name' to 'new_module_name'
        if module == 'scLTMF':
            module = 'scTrace.scLTMF'
        return super().find_class(module, name)


def load_model(filename):
    with open(filename, 'rb') as f:
        return CustomUnpickler(f).load()