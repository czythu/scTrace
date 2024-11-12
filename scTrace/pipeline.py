import gc
from .scLTMF import scLTMF
from .utils import *
import scStateDynamics as scd


def prepareCrosstimeGraph(data_pre, data_pos, lineage_identity, pre_name, pos_name,
                            savePath, run_label_time):
    cross_sim = getSimilarityMatrix(data_pre, data_pos, method = 'Pearson')
    barcodes_pre, barcodes_pos = list(data_pre.obs[lineage_identity]), list(data_pos.obs[lineage_identity])
    cross_lin_mat, clonotype_mat = getLineageMatrix(bars = barcodes_pre, bars2 = barcodes_pos)
    getCrossLineageDensity(cross_lin_mat)
    cross_sp_df = np.multiply(cross_sim, cross_lin_mat)
    cross_sp_df, clonotype_df = coo_matrix(cross_sp_df), coo_matrix(clonotype_mat)
    cross_sp_df = pd.DataFrame({'u_id':cross_sp_df.row, 'i_id':cross_sp_df.col, 'flow':cross_sp_df.data, 'clonotype':clonotype_df.data})
    cross_sp_df.to_csv(savePath + run_label_time + '-cross_df.csv')
    del(clonotype_df, clonotype_mat)
    gc.collect()
    print("Generating mother-daughter similarity")
    plotSimilarityCompare(cross_sim=cross_sim,
                          cross_lin_mat=cross_lin_mat,
                          title=pre_name + ' -> ' + pos_name,
                          savePath=savePath + run_label_time + '-SimDistrComp-hist.png')

    return cross_sp_df, cross_lin_mat, barcodes_pre, barcodes_pos


def prepareWithintimeGraph(data_pre, data_pos, lineage_identity, pre_name, pos_name, savePath, run_label_time, mode="pre"):
    if mode == "pre":
        data, title = data_pre, pre_name
    else:
        data, title = data_pos, pos_name
    sim_mat = getSimilarityMatrix(data, data, method = 'Pearson')
    barcodes = list(data.obs[lineage_identity])
    within_lin_mat, _ = getLineageMatrix(bars = barcodes, bars2 = barcodes)
    print("Generating mother-daughter similarity (with-in timepoint)")
    plotSimilarityCompare(cross_sim=sim_mat, cross_lin_mat=within_lin_mat, title=title,
                          savePath=savePath + run_label_time + '-SimDistrComp-hist-'+ mode +'.png')

    return sim_mat, within_lin_mat


# Generate side information
def prepareSideInformation(data_pre, data_pos, barcodes_pre, barcodes_pos,
                           savePath, run_label_time, single_inte_fraction=0.5):
    pre_lin_mat, _ = getLineageMatrix(bars = barcodes_pre)
    pos_lin_mat, _ = getLineageMatrix(bars = barcodes_pos)
    pre_sim_mat = single_inte_fraction * np.array(data_pre.obsp['connectivities'].todense()) + (1 - single_inte_fraction) * pre_lin_mat
    pos_sim_mat = single_inte_fraction * np.array(data_pos.obsp['connectivities'].todense()) + (1 - single_inte_fraction) * pos_lin_mat
    Ku_inv = inv_node2vec_kernel(pre_sim_mat)
    np.save(savePath + run_label_time + '-Ku_inv.npy', Ku_inv)
    Kv_inv = inv_node2vec_kernel(pos_sim_mat)
    np.save(savePath + run_label_time + '-Kv_inv.npy', Kv_inv)
    del(pre_sim_mat, pos_sim_mat)
    gc.collect()

# epoch in train
def trainMF(train_df, val_df, n_pre, n_pos, savePath, run_label_time,
            n_factor=20, n_epoch=400, bool_pre_side=True, bool_post_side=True,
            learning_rate=0.01, regularization=0.0001):
    print("Loading side information")
    Ku_inv = np.load(savePath + run_label_time + '-Ku_inv.npy')
    Kv_inv = np.load(savePath + run_label_time + '-Kv_inv.npy')
    print("Performing matrix factorization")
    hyper_dict, model = grid_search(scLTMF, train_df.iloc[:,:3], val_df.iloc[:,:3], n_pre, n_pos, n_factor, n_epoch,
                                    Ku_inv, Kv_inv, bool_pre_side=bool_pre_side, bool_post_side=bool_post_side,
                                    learning_rate=learning_rate, regularization=regularization)
    # Ablation experiment
    if bool_pre_side == True and bool_post_side == False:
        run_label_time = run_label_time + '_keepSu'
    elif bool_pre_side == False and bool_post_side == True:
        run_label_time = run_label_time + '_keepSv'
    elif bool_pre_side == False and bool_post_side == False:
        run_label_time = run_label_time + '_NoSide'
    # Keep all side information
    else:
        print("Saving model")
        f = open(savePath + run_label_time + '_model.pkl', 'wb')
        pickle.dump(model, f)
        f.close()

    print("Saving train results")
    val_recall, train_recall = model.list_val_recall, model.list_train_recall
    val_rmse, train_rmse = model.list_val_rmse, model.list_train_rmse
    training_results = [val_recall, train_recall, val_rmse, train_rmse]
    np.save(savePath + run_label_time + '.npy', training_results)

    plot_metrics(model, savePath, run_label_time, run_label_time)

    return hyper_dict, model


def predictMissingEntries(pre_name, pos_name, savePath, run_label_time, showName, threshold_positive=0.25):
    print("Loading pretrained model...")
    # with open(savePath + run_label_time + '_model.pkl', 'rb') as file:
    #     model = pickle.load(file)
    model = load_model(savePath + run_label_time + '_model.pkl')
    min_rmse, max_recall = plot_metrics(model, savePath, run_label_time, showName + ': ' + pre_name + '->' + pos_name)
    y_true = model.train[:, 2]
    y_true_val = model.val[:, 2]
    pred_mat = np.dot(model.p, model.q.T)
    y_pred = np.array([pred_mat[int(model.train[i, 0]), int(model.train[i, 1])] for i in range(model.train.shape[0])])
    y_pred_val = np.array([pred_mat[int(model.val[i, 0]), int(model.val[i, 1])] for i in range(model.val.shape[0])])
    # threshold = threshold_positive
    threshold = min(np.max(model.train[:, 2]) / 2, np.mean(model.train[:, 2]) - 2 * np.std(model.train[:, 2]))
    complet_mat = np.dot(model.p, model.q.T)
    complet_mat[complet_mat < threshold] = 0
    corr = plotFittingResults(y_pred, y_true, y_pred_val, y_true_val,
                              savePath, run_label_time, showName + ': ' + pre_name + '->' + pos_name)
    return pred_mat, y_true, y_pred, complet_mat, corr, min_rmse, max_recall


def prepareScdobj(data_pre, data_pos, time, pre_name, pos_name, cls_res_all, clq_res_all,
                  pre_colors, pos_colors, savePath, run_label_time):
    cls_res_pre, clq_res_pre = cls_res_all[time], clq_res_all[time]
    cls_res_pos, clq_res_pos = cls_res_all[time+1], clq_res_all[time+1]
    scd_obj = scd.scStateDynamics(data_pre = data_pre, data_pos = data_pos, pre_name = pre_name, pos_name = pos_name,
                                  cls_prefixes = ['', ''], run_label = run_label_time, pre_colors = pre_colors,
                                  pos_colors = pos_colors, savePath = savePath, saveFigFormat = "png")
    scd_obj.runClustering(cls_resolutions = [cls_res_pre, cls_res_pos], clq_resolutions = [clq_res_pre, clq_res_pos])

    return scd_obj


def visualizeLineageInfo(scd_obj, cross_lin_mat, n_pre, pre_colors, pos_colors,
                         pre_name, pos_name, savePath, run_label_time):
    pre_fractions, pos_fractions = calculateFractions(scd_obj)
    t_row_sum = cross_lin_mat.sum(axis=1, keepdims=True)
    t_row_sum[t_row_sum == 0] = 1
    cls_lineage_mat = cross_lin_mat / t_row_sum
    cls_lineage_mat = np.array([[np.sum(cls_lineage_mat[np.where(scd_obj.data_pre.obs['cluster'] == str(i))[0]][:,
                                        np.where(scd_obj.data_pos.obs['cluster'] == str(j))[0]])
                                 for j in range(scd_obj.n_clus[1])] for i in range(scd_obj.n_clus[0])])
    cls_lineage_mat = cls_lineage_mat / n_pre
    flow_info = {'s': [str(i) for i in range(scd_obj.n_clus[0]) for j in range(scd_obj.n_clus[1])],
                 't': [str(j) for i in range(scd_obj.n_clus[0]) for j in range(scd_obj.n_clus[1])],
                 's_pm': list(cls_lineage_mat.reshape((1, -1))[0]),
                 't_pm': list(cls_lineage_mat.reshape((1, -1))[0])}
    flow_info = pd.DataFrame(flow_info)
    fig = plotFlowSankey(flow_info, pre_colors, pos_colors, pre_fractions=pre_fractions, pos_fractions=pos_fractions,
                         figwidth = 3.62, figheight = 6, label_size = 18, title = pre_name + '->' + pos_name,
                         label_position='twoside')
    fig.savefig(savePath + run_label_time + "-FlowSankey-lineage.png", dpi=300)
    fig.savefig(savePath + run_label_time + "-FlowSankey-lineage.pdf", dpi=300)

    return cls_lineage_mat, flow_info


def visualizeEnhancedLineageInfo(scd_obj, complet_mat, n_pre, pre_colors, pos_colors,
                                 pre_name, pos_name, savePath, run_label_time):
    pre_fractions, pos_fractions = calculateFractions(scd_obj)
    # cell_trans_mat = 0 + (complet_mat > 0)
    cell_trans_mat = complet_mat
    t_row_sum = cell_trans_mat.sum(axis=1, keepdims=True)
    t_row_sum[t_row_sum == 0] = 1
    cell_trans_mat = cell_trans_mat / t_row_sum
    cls_trans_mat = np.array([[np.sum(cell_trans_mat[np.where(scd_obj.data_pre.obs['cluster'] == str(i))[0]][:, np.where(scd_obj.data_pos.obs['cluster'] == str(j))[0]]) for j in range(scd_obj.n_clus[1])] for i in range(scd_obj.n_clus[0])])
    cls_trans_mat = cls_trans_mat / n_pre

    flow_info = {'s':[str(i) for i in range(scd_obj.n_clus[0]) for j in range(scd_obj.n_clus[1])],
                 't':[str(j) for i in range(scd_obj.n_clus[0]) for j in range(scd_obj.n_clus[1])],
                 's_pm':list(cls_trans_mat.reshape((1,-1))[0]),
                 't_pm':list(cls_trans_mat.reshape((1,-1))[0])}
    flow_info = pd.DataFrame(flow_info)

    fig = plotFlowSankey(flow_info, pre_colors, pos_colors, pre_fractions=pre_fractions, pos_fractions=pos_fractions,
                         figwidth = 3.62, figheight = 6, label_size = 18, title = pre_name + '->' + pos_name,
                         label_position='twoside')
    fig.savefig(savePath + run_label_time + "-FlowSankey-complete.png", dpi=300)
    fig.savefig(savePath + run_label_time + "-FlowSankey-complete.pdf", dpi=300)
    return cls_trans_mat, flow_info


def assignLineageInfo(scd_obj, cross_lin_mat, savePath, run_label_time, sel_cluster_name="cluster"):
    pos_celltypes = scd_obj.data_pos.obs[sel_cluster_name]
    cell_2lin_cls = np.array([(cross_lin_mat[:, np.where(pos_celltypes == celltype)[0]] > 0).sum(axis=1).tolist()
                              for celltype in np.unique(pos_celltypes)]).T

    fate_cls = np.unique(pos_celltypes)[np.argmax(cell_2lin_cls, axis=1)]  # .astype(str)
    fate_cls[np.sum(cell_2lin_cls, axis=1) == 0] = 'Missing'
    scd_obj.data_pre.obs['Lineage_fate'] = fate_cls

    fate_cls = scd_obj.data_pre.obs[sel_cluster_name].astype(str) + ' -> ' + fate_cls
    fate_cls[np.sum(cell_2lin_cls, axis=1) == 0] = 'Missing'
    scd_obj.data_pre.obs['Lineage_fate_label'] = fate_cls
    plotCellFate(scd_obj.data_pre, savePath, run_label_time, cls_colname=sel_cluster_name, fate_colname='Lineage_fate_label',
                 special_case="Missing", png_name="_cellfate-umap-onlyLT.png")
    return scd_obj


def enhanceFate(scd_obj, complete_mat, savePath, run_label_time,
                cluster_name="cluster", lineage_fate_colname = 'Lineage_fate',
                enhanced_fate_colname = 'Enhanced_fate', cutoff=True, method="ranksum"):
    adata_pre, adata_pos = scd_obj.data_pre, scd_obj.data_pos
    # Select non-zero values from completed matrix
    pos_celltypes = np.unique(adata_pos.obs[cluster_name])
    values_nz = [[row[row != 0] for row in complete_mat[:, adata_pos.obs[cluster_name] == pos_celltype]]
                 for pos_celltype in pos_celltypes]

    cell_fate_cls = []
    num_pos_clusters = len(values_nz)
    for ii in trange(complete_mat.shape[0]):
        # number of cells in pre-data
        # print(ii)
        # non-zero values in transition matrix for all clusters
        cell_ii_values = [values_nz[jj][ii] for jj in range(num_pos_clusters)]
        cur_res = 'Uncertain'
        # number of clusters in post-data
        p_value_list = []
        for jj in range(num_pos_clusters):
            cur_values = cell_ii_values[jj]
            if len(cur_values) > 0:
                # other_vals = [x for j2 in range(num_pos_clusters) for x in values_nz[j2][jj] if j2 != 0]
                other_vals = np.array([x for j2 in range(num_pos_clusters) for x in cell_ii_values[j2] if j2 != jj])
                if len(other_vals) == 0:
                    # cur_res = str(jj)
                    cur_res = pos_celltypes[jj]
                    break
                else:
                    cur_p_value = 1
                    if method == "ranksum":
                        '''
                        Compute the Wilcoxon rank-sum statistic for two samples.
                        The Wilcoxon rank-sum test tests the null hypothesis that two sets of measurements are drawn
                        from the same distribution.
                        The alternative hypothesis is that values in one sample are more likely to be larger than the values
                        in the other sample.
                        '''
                        cur_statistic, cur_p_value = ranksums(cur_values, other_vals, alternative='greater')
                    elif method == "meanwhit":
                        '''
                        Perform the Mann-Whitney U rank test on two independent samples.
                        The Mann-Whitney U test is a nonparametric test of the null hypothesis that the distribution
                        underlying sample x is the same as the distribution underlying sample y.
                        It is often used as a test of difference in location between distributions.
                        '''
                        cur_statistic, cur_p_value = mannwhitneyu(cur_values, other_vals, use_continuity=True, alternative='greater')
                    else:
                        print("Please choose method from ['ranksum', 'meanwhit']")
                    p_value_list.append(cur_p_value)

        if len(p_value_list) > 0:
            if cutoff == True:
                if np.min(p_value_list) < 0.05:
                    cur_res = pos_celltypes[np.argmin(np.array(p_value_list))]
            else:
                cur_res = pos_celltypes[np.argmin(np.array(p_value_list))]

        cell_fate_cls.append(cur_res)


    adata_pre.obs[enhanced_fate_colname] = adata_pre.obs[lineage_fate_colname].astype('str')

    # Supplement cell fate relationships that have not been captured by lineage-tracing
    count_offtarget, count_enhance = 0, 0
    for i in range(len(adata_pre.obs[enhanced_fate_colname])):
        if adata_pre.obs[enhanced_fate_colname][i] == "Missing":
            count_offtarget += 1
            adata_pre.obs[enhanced_fate_colname][i] = cell_fate_cls[i]
            if cell_fate_cls[i] != "Uncertain":
                count_enhance += 1
    enhance_rate = count_enhance / count_offtarget
    print("Ratio of newly added fate clusters: {:.4f}".format(enhance_rate))

    # scd_obj.data_pre.obs['Fate_cls'] = cell_fate_cls
    adata_pre.obs[cluster_name] = adata_pre.obs[cluster_name].astype('str')
    adata_pre.obs[enhanced_fate_colname+'_label'] = adata_pre.obs[cluster_name] + ' -> ' + adata_pre.obs[enhanced_fate_colname]
    adata_pre.obs.loc[adata_pre.obs[enhanced_fate_colname+'_label'].str.contains('Uncertain', case=False), enhanced_fate_colname+'_label'] = 'Uncertain'

    plotCellFate(adata_pre, savePath, run_label_time, cls_colname='Cell type annotation',
                 fate_colname='Enhanced_fate_label',
                 special_case="Uncertain", png_name="_cellfate-umap-enhanced.png")

    return adata_pre, enhance_rate


def runFateDE(adata_pre, fate_colname, sel_cls, sel_fates, saveName, filter_signif=True):
    all_de_df = pd.DataFrame()
    cur_expr = adata_pre[adata_pre.obs[fate_colname].isin(sel_fates)]

    t_v, t_n = np.unique(cur_expr.obs[fate_colname], return_counts=True)

    if np.all(t_n > 1) == False:
        mask = np.isin(np.array(cur_expr.obs[sel_fates]), t_v[np.argwhere(t_n > 1).flatten()])
        cur_expr = cur_expr[np.where(mask)[0]]
        t_v, t_n = np.unique(cur_expr.obs[sel_fates], return_counts=True)

    try:
        # cur_expr.uns['log1p']["base"] = None
        sc.tl.rank_genes_groups(cur_expr, groupby=fate_colname, method='wilcoxon')
        subfates = [name for name, _ in cur_expr.uns['rank_genes_groups']['names'].dtype.fields.items()]
        for i in range(len(subfates)):
            de_df = generateDEGs(cur_expr, index=i, cluster=sel_cls, fate_str=subfates[i], filter_signif=filter_signif)
            all_de_df = pd.concat([all_de_df, de_df], axis=0)

    except ZeroDivisionError as e:
        print(e)

    if all_de_df.shape[0] > 0:
        all_de_df.to_csv(saveName, sep='\t', header=True, index=False)

    return cur_expr, all_de_df


def dynamicDiffAnalysis(scd_obj, savePath, run_label_time,
                        sel_cluster_name="cluster", fate_colname='Lineage_fate', special_case="Missing"):
    all_de_df = pd.DataFrame()
    pre_celltypes = np.unique(scd_obj.data_pre.obs[sel_cluster_name])
    for si in pre_celltypes:
        cur_expr = scd_obj.data_pre[scd_obj.data_pre.obs[sel_cluster_name] == str(si)]
        cur_expr = cur_expr[cur_expr.obs[fate_colname] != special_case]

        t_v, t_n = np.unique(cur_expr.obs[fate_colname], return_counts=True)
        if np.all(t_n > 1) == False:
            mask = np.isin(np.array(cur_expr.obs[fate_colname]), t_v[np.argwhere(t_n > 1).flatten()])
            cur_expr = cur_expr[np.where(mask)[0]]
            t_v, t_n = np.unique(cur_expr.obs[fate_colname], return_counts=True)
        try:
            sc.tl.rank_genes_groups(cur_expr, groupby=fate_colname, method='wilcoxon')
            # Select genes
            for i, f in enumerate(cur_expr.obs[fate_colname].unique()):
                fate_str = str(si) + '->' + str(f)
                de_df = generateDEGs(cur_expr, index=i, cluster=si, fate_str=fate_str)
                all_de_df = pd.concat([all_de_df, de_df], axis=0)

        except ZeroDivisionError as e:
            print(e)

    if all_de_df.shape[0] > 0:
        if fate_colname == "Lineage_fate":
            all_de_df.to_csv(savePath + run_label_time + '_DE_all_fate_genes-onlyLT.txt', sep='\t', header=True, index=False)
        else:
            all_de_df.to_csv(savePath + run_label_time + '_DE_all_fate_genes-enhanced.txt', sep='\t', header=True, index=False)
    return all_de_df

