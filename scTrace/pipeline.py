import gc
from .scLTMF import scLTMF
from .utils import *
import scStateDynamics as scd


def prepareCrosstimeGraph(data_pre, data_pos, lineage_identity, pre_name, pos_name,
                            savePath, run_label_time):
    cross_sim = getSimilarityMatrix(data_pre, data_pos, method = 'Pearson')
    # lineage_identity = 'Clone' in LARRY
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
    # lineage_identity = 'Clone' in LARRY
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
    # np.save(savePath + run_label_time + '-Ku_inv_down50.npy', Ku_inv)
    np.save(savePath + run_label_time + '-Ku_inv.npy', Ku_inv)
    Kv_inv = inv_node2vec_kernel(pos_sim_mat)
    # np.save(savePath + run_label_time + '-Kv_inv_down50.npy', Kv_inv)
    np.save(savePath + run_label_time + '-Kv_inv.npy', Kv_inv)
    del(pre_sim_mat, pos_sim_mat)
    gc.collect()

# epoch in train
def trainMF(train_df, val_df, n_pre, n_pos, savePath, run_label_time,
            n_factor=20, n_epoch=400, bool_pre_side=True, bool_post_side=True):
    # Ku_inv = np.load(savePath + run_label_time + '-Ku_inv_down50.npy')
    # Kv_inv = np.load(savePath + run_label_time + '-Kv_inv_down50.npy')
    print("Loading side information")    # (lr, reg) in grid_research
    # Watermelon: (0.01, 0.0001), TraceSeq: (0.01, 0.0001), Larry in vitro day4-6: (0.01, 0.0001)
    # C-elegans 300min-400min: (0.01, 0.0001), 400min-500min: (0.01, 0.00005)
    # CellTagging: (0.01, 0.0001), JMML-TCR: (0.01, 0.0001)
    Ku_inv = np.load(savePath + run_label_time + '-Ku_inv.npy')
    Kv_inv = np.load(savePath + run_label_time + '-Kv_inv.npy')
    # (lr, reg) in grid_research
    # Watermelon: (0.01, 0.0001), TraceSeq: (0.01, 0.0001), Larry in vitro day4-6: (0.01, 0.0001)
    # C-elegans 300min-400min: (0.01, 0.0001), 400min-500min: (0.01, 0.00005)
    # CellTagging: (0.01, 0.0001), JMML-TCR: (0.01, 0.0001)
    print("Performing matrix factorization")
    hyper_dict, model = grid_search(scLTMF, train_df.iloc[:,:3], val_df.iloc[:,:3], n_pre, n_pos, n_factor, n_epoch,
                                    Ku_inv, Kv_inv, bool_pre_side=bool_pre_side, bool_post_side=bool_post_side)
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
    threshold = threshold_positive
    complet_mat = np.dot(model.p, model.q.T)
    complet_mat[complet_mat < threshold] = 0
    corr = plotFittingResults(pred_mat, y_pred, y_true, y_pred_val, y_true_val,
                              pre_name, pos_name, savePath, run_label_time, showName + ': ' + pre_name + '->' + pos_name)
    return pred_mat, y_true, y_pred, complet_mat, corr, min_rmse, max_recall


def prepareScdobj(data_pre, data_pos, time, pre_name, pos_name, cls_res_all, clq_res_all,
                  pre_colors, pos_colors, savePath, run_label_time):
    cls_res_pre, clq_res_pre = cls_res_all[time], clq_res_all[time]
    cls_res_pos, clq_res_pos = cls_res_all[time+1], clq_res_all[time+1]
    scd_obj = scd.scStateDynamics(data_pre = data_pre, data_pos = data_pos, pre_name = pre_name, pos_name = pos_name,
                                  cls_prefixes = ['', ''], run_label = run_label_time, pre_colors = pre_colors,
                                  pos_colors = pos_colors, savePath = savePath, saveFigFormat = "png")
    # scd_obj = scStateDynamics(data_pre = data_pre, data_pos = data_pos, pre_name = pre_name, pos_name = pos_name,
    #                               cls_prefixes = ['', ''], run_label = run_label_time, pre_colors = pre_colors,
    #                               pos_colors = pos_colors, savePath = savePath, saveFigFormat = "png")
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


def assignLineageInfo(scd_obj, cross_lin_mat, savePath, run_label_time):
    cell_2lin_cls = np.array([(cross_lin_mat[:,
                               np.where(scd_obj.data_pos.obs['cluster'] == str(j))[0]] > 0).sum(axis=1).tolist()
                              for j in range(scd_obj.n_clus[1])]).T
    fate_cls = np.argmax(cell_2lin_cls, axis=1).astype(str)
    fate_cls[np.sum(cell_2lin_cls, axis=1) == 0] = 'offTarget'
    scd_obj.data_pre.obs['Fate_cls'] = fate_cls
    fate_cls = scd_obj.data_pre.obs['cluster'].astype(str) + '->' + fate_cls
    fate_cls[np.sum(cell_2lin_cls, axis=1) == 0] = 'offTarget'
    scd_obj.data_pre.obs['Fate'] = fate_cls
    print(np.unique(scd_obj.data_pre.obs['Fate'], return_counts=True))
    # cfrs, afs = compute_fate_vector(scd_obj, cross_lin_mat)
    cfrs, afs = compute_fate_vector(scd_obj.data_pre, cell_2lin_cls)
    summary_metric, ncs, jics, ecs = calculateFateDiversity(scd_obj.data_pre)
    plotCellFate(scd_obj.data_pre, savePath, run_label_time)
    with plt.rc_context({'figure.figsize': (3, 3)}):
        color_temp = ["#43D9FE", "#E78AC3", "#FEC643", "#A6D854","#FE6943", "#E5C494", "#33AEB1", "#FFEC1A","#66C2A5", "#FC8D62"]
        sc.pl.umap(scd_obj.data_pos, color="cluster", palette=color_temp, show=False)
    plt.savefig(savePath + run_label_time + '_pos_cluster.png', dpi=300, bbox_inches='tight')
    return scd_obj, summary_metric, ncs, jics, ecs, cfrs, afs


def enhanceLineageInfo(scd_obj, cross_lin_mat, complet_mat):
    cell_2lin_cls = np.array([(cross_lin_mat[:,
                               np.where(scd_obj.data_pos.obs['cluster'] == str(j))[0]] > 0).sum(axis=1).tolist() for j
                              in range(scd_obj.n_clus[1])]).T
    fate_cls = np.argmax(cell_2lin_cls, axis=1).astype(str)
    fate_cls[np.sum(cell_2lin_cls, axis=1) == 0] = 'offTarget'
    scd_obj.data_pre.obs['Fate_cls'] = fate_cls
    scd_obj, enhance_rate = assignFate(scd_obj, complete_mat=complet_mat)
    # print(np.unique(scd_obj.data_pre.obs['Fate'], return_counts=True))
    summary_metric, ncs, jics, ecs = calculateFateDiversity(scd_obj.data_pre)
    return scd_obj, summary_metric, ncs, jics, ecs


def originalDynamicAnalysis(scd_obj, savePath, run_label_time):
    all_de_df = pd.DataFrame()
    for si in range(scd_obj.n_clus[0]):
        cur_expr = scd_obj.data_pre[scd_obj.data_pre.obs['cluster'] == str(si)]
        cur_expr = cur_expr[cur_expr.obs['Fate_cls'] != 'offTarget']

        t_v, t_n = np.unique(cur_expr.obs['Fate_cls'], return_counts=True)
        print(t_v, t_n)
        if np.all(t_n > 1) == False:
            # mask = np.isin(np.array(cur_expr.obs['Fate_cls']), np.argwhere(t_n > 1).flatten().astype(str))
            mask = np.isin(np.array(cur_expr.obs['Fate_cls']), t_v[np.argwhere(t_n > 1).flatten()])
            cur_expr = cur_expr[np.where(mask)[0]]
            t_v, t_n = np.unique(cur_expr.obs['Fate_cls'], return_counts=True)
            print(t_v, t_n)
        try:
            sc.tl.rank_genes_groups(cur_expr, groupby='Fate_cls', method='wilcoxon')
            # Select genes
            for i, f in enumerate(cur_expr.obs['Fate_cls'].unique()):
                fate_str = str(si) + '->' + str(f)
                de_df = generateDEGs(cur_expr, index=i, cluster=si, fate_str=fate_str)
                print(de_df.shape)
                all_de_df = pd.concat([all_de_df, de_df], axis=0)

        except ZeroDivisionError as e:
            print(e)

    if all_de_df.shape[0] > 0:
        all_de_df.to_csv(savePath + run_label_time + '_DE_fate_genes-S0-T0_onlyLT.txt',
                         sep='\t', header=True, index=False)

    p_data = pd.DataFrame({'UMAP_1': scd_obj.data_pre.obsm['X_umap'][:, 0],
                           'UMAP_2': scd_obj.data_pre.obsm['X_umap'][:, 1],
                           'Cluster': scd_obj.data_pre.obs['cluster'],
                           'Fate': list(scd_obj.data_pre.obs['Fate'])})
    p_data = p_data.sort_values(by='Fate', axis=0, ascending=True)
    p_data.to_csv(savePath + run_label_time + '_lineageFate_df.txt', sep='\t', header=True, index=False)

    # plotCellFate(scd_obj.data_pre, savePath, run_label_time)

    return all_de_df


def enhancedDynamicAnalysis(scd_obj, savePath, run_label_time):
    all_de_df = pd.DataFrame()
    for si in range(scd_obj.n_clus[0]):
        cur_expr = scd_obj.data_pre[scd_obj.data_pre.obs['cluster'] == str(si)]
        cur_expr = cur_expr[cur_expr.obs['Fate_cls'] != 'Uncertain']
        t_v, t_n = np.unique(cur_expr.obs['Fate_cls'], return_counts=True)
        print(t_v, t_n)
        if np.all(t_n > 1) == False:
            # mask = np.isin(np.array(cur_expr.obs['Fate_cls']), np.argwhere(t_n > 1).flatten().astype(str))
            mask = np.isin(np.array(cur_expr.obs['Fate_cls']), t_v[np.argwhere(t_n > 1).flatten()])
            cur_expr = cur_expr[np.where(mask)[0]]
            t_v, t_n = np.unique(cur_expr.obs['Fate_cls'], return_counts=True)
            print(t_v, t_n)
        try:
            sc.tl.rank_genes_groups(cur_expr, groupby='Fate_cls', method='wilcoxon')
            # Select genes
            for i, f in enumerate(cur_expr.obs['Fate_cls'].unique()):
                fate_str = str(si) + '->' + str(f)
                de_df = generateDEGs(cur_expr, index=i, cluster=si, fate_str=fate_str)
                print(de_df.shape)
                all_de_df = pd.concat([all_de_df, de_df], axis=0)

        except ZeroDivisionError as e:
            print(e)

    if all_de_df.shape[0] > 0:
        all_de_df.to_csv(savePath + run_label_time + '_DE_fate_genes-S0-T0_1.txt', sep='\t', header=True, index=False)

    p_data = pd.DataFrame({'UMAP_1': scd_obj.data_pre.obsm['X_umap'][:, 0],
                           'UMAP_2': scd_obj.data_pre.obsm['X_umap'][:, 1],
                           'Cluster': scd_obj.data_pre.obs['cluster'],
                           'Fate': list(scd_obj.data_pre.obs['Fate_cls'])})
    p_data = p_data.sort_values(by='Fate', axis=0, ascending=True)
    p_data.to_csv(savePath + run_label_time + '_predFate_df.txt', sep='\t', header=True, index=False)

    plotCellFate(scd_obj.data_pre, savePath, run_label_time,
                 special_case=tuple(np.unique(scd_obj.data_pre.obs["cluster"]) + "->Uncertain"),
                 png_name='_cellfate-umap-aftermc.png')

    return all_de_df