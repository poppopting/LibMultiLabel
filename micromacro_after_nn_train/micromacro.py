import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict

def train_thresholding(y, yhat, verbose=True):

    num_class = y.shape[1]
    thresholds = np.zeros(num_class)

    if verbose:
        # logging.info("Training thresholding model on %s labels", num_class)
        print("Training thresholding model on %s labels" % num_class)

    num_positives = np.sum(y, 0) # np.sum(y, 2)
    label_order = np.flip(np.argsort(num_positives)).flat

    # accumulated counts for micro
    stats = {"tp": 0, "fp": 0, "fn": 0, "labels": 0}

    for i in tqdm(label_order, disable=not verbose):
        yi = y[:, i].reshape(-1)
        yi_hat = yhat[:, i].reshape(-1)
        t, stats = _micromacro_one_label(yi, yi_hat, stats)
        thresholds[i] = t

    return thresholds

def _micromacro_one_label(y, yhat, stats):

    thresholds = 0

    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    stats["labels"] += 1

    def micro_plus_macro(tp, fp, fn):
        # Because the F-measure of other labels are constants and thus does not affect optimization,
        # we ignore them when calculating macro-F.
        macro = np.nan_to_num((2 * tp) / (2 * tp + fp + fn)) / stats["labels"]
        micro = np.nan_to_num((2 * (tp + stats["tp"])) / (2 * (tp + stats["tp"]) + fp + fn + stats["fp"] + stats["fn"]))
        return micro + macro

    sorted_yhat_index = np.argsort(yhat, kind="stable")
    sorted_yhat = yhat[sorted_yhat_index]

    # ignore warning for 0/0 when calculating F-measures
    prev_settings = np.seterr("ignore")

    tp = np.sum(y == 1)
    fp = y.size - tp
    fn = 0
    best_obj = micro_plus_macro(tp, fp, fn)
    best_tp, best_fp, best_fn = tp, fp, fn
    cut = -1

    for i in range(y.size):
        if y[sorted_yhat_index[i]] == 0:
            fp -= 1
        else:
            tp -= 1
            fn += 1

        obj = micro_plus_macro(tp, fp, fn)

        if obj >= best_obj:
            best_obj = obj
            best_tp, best_fp, best_fn = tp, fp, fn
            cut = i
    np.seterr(**prev_settings)

    if cut == -1:  # i.e. all 1 in scut
        thresholds = np.nextafter(sorted_yhat[0], -np.inf)  # predict all 1
    elif cut == y.size - 1:
        thresholds = np.nextafter(sorted_yhat[-1], np.inf)
    else:
        thresholds = (sorted_yhat[cut] + sorted_yhat[cut + 1]) / 2

    tp_sum += best_tp
    fp_sum += best_fp
    fn_sum += best_fn

    # # In FlatModel.predict_values, the threshold is added to the decision value.
    # # Therefore, we need to make it negative here.
    # threshold = -thresholds

    threshold = thresholds
    stats["tp"] += tp_sum
    stats["fp"] += fp_sum
    stats["fn"] += fn_sum

    return threshold, stats

def sigmoid(x):
    return np.where(
        x >= 0, # condition
        1 / (1 + np.exp(-x)), # For positive values
        np.exp(x) / (1 + np.exp(x)) # For negative values
    )


def train_thresholding_folds(y_list:list, yhat_list:list, verbose=True):


    num_class = y_list[0].shape[1]
    thresholds = np.zeros(num_class)

    if verbose:
        # logging.info("Training thresholding model on %s labels", num_class)
        print("Training thresholding model on %s labels" % num_class)

    num_positives = np.sum(y_list[0], 0) # np.sum(y, 2)
    label_order = np.flip(np.argsort(num_positives)).flat

    # accumulated counts for micro
    stats = {"tp": 0, "fp": 0, "fn": 0, "labels": 0}

    for i in tqdm(label_order, disable=not verbose):
        yi = [y[:, i].reshape(-1) for y in y_list] #y[:, i].reshape(-1)
        yi_hat = [yhat[:, i].reshape(-1) for yhat in yhat_list]#yhat[:, i].reshape(-1)
        t, stats = _micromacro_one_label_folds(yi, yi_hat, stats)
        thresholds[i] = t

    return thresholds

def _micromacro_one_label_folds(y_list, yhat_list, stats):

    nr_fold = len(y_list) #5
    thresholds = np.zeros(nr_fold)

    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    stats["labels"] += 1

    def micro_plus_macro(tp, fp, fn):
        # Because the F-measure of other labels are constants and thus does not affect optimization,
        # we ignore them when calculating macro-F.
        macro = np.nan_to_num((2 * tp) / (2 * tp + fp + fn)) / stats["labels"]
        micro = np.nan_to_num((2 * (tp + stats["tp"])) / (2 * (tp + stats["tp"]) + fp + fn + stats["fp"] + stats["fn"]))
        return micro + macro

    for fold in range(nr_fold):

        y = y_list[fold]
        yhat = yhat_list[fold]
        sorted_yhat_index = np.argsort(yhat, kind="stable")
        sorted_yhat = yhat[sorted_yhat_index]

        # ignore warning for 0/0 when calculating F-measures
        prev_settings = np.seterr("ignore")

        tp = np.sum(y == 1)
        fp = y.size - tp
        fn = 0
        best_obj = micro_plus_macro(tp, fp, fn)
        best_tp, best_fp, best_fn = tp, fp, fn
        cut = -1

        for i in range(y.size):
            if y[sorted_yhat_index[i]] == 0:
                fp -= 1
            else:
                tp -= 1
                fn += 1

            obj = micro_plus_macro(tp, fp, fn)

            if obj >= best_obj:
                best_obj = obj
                best_tp, best_fp, best_fn = tp, fp, fn
                cut = i
        np.seterr(**prev_settings)

        if cut == -1:  # i.e. all 1 in scut
            thresholds[fold] = np.nextafter(sorted_yhat[0], -np.inf)  # predict all 1
        elif cut == y.size - 1:
            thresholds[fold] = np.nextafter(sorted_yhat[-1], np.inf)
        else:
            thresholds[fold] = (sorted_yhat[cut] + sorted_yhat[cut + 1]) / 2

        tp_sum += best_tp
        fp_sum += best_fp
        fn_sum += best_fn

    # # In FlatModel.predict_values, the threshold is added to the decision value.
    # # Therefore, we need to make it negative here.
    # threshold = -thresholds

    threshold = thresholds.mean()
    stats["tp"] += tp_sum
    stats["fp"] += fp_sum
    stats["fn"] += fn_sum

    return threshold, stats

def train_thresholding_v2(y, yhat, candidate, verbose=True):
    '''use for the scenerio that you want to fit micromacro on one set but using decision values from the other set'''
    num_class = y.shape[1]
    thresholds = np.zeros(num_class)

    if verbose:
        # logging.info("Training thresholding model on %s labels", num_class)
        print("Training thresholding model on %s labels" % num_class)

    num_positives = np.sum(y, 0) # np.sum(y, 2)
    label_order = np.flip(np.argsort(num_positives)).flat

    # accumulated counts for micro
    stats = {"tp": 0, "fp": 0, "fn": 0, "labels": 0}

    for i in tqdm(label_order, disable=not verbose):
        yi = y[:, i].reshape(-1)
        yi_hat = yhat[:, i].reshape(-1)
        yi_candidate = candidate[:, i].reshape(-1)
        t, stats = _micromacro_one_label_v2(yi, yi_hat, yi_candidate, stats)
        thresholds[i] = t

    return thresholds

def _micromacro_one_label_v2(y, yhat, candidate, stats):

    thresholds = 0

    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    stats["labels"] += 1

    def micro_plus_macro(tp, fp, fn):
        # Because the F-measure of other labels are constants and thus does not affect optimization,
        # we ignore them when calculating macro-F.
        macro = np.nan_to_num((2 * tp) / (2 * tp + fp + fn)) / stats["labels"]
        micro = np.nan_to_num((2 * (tp + stats["tp"])) / (2 * (tp + stats["tp"]) + fp + fn + stats["fp"] + stats["fn"]))
        return micro + macro

    sorted_candidate_index = np.argsort(candidate, kind="stable")
    sorted_candidate = candidate[sorted_candidate_index]

    sorted_yhat_index = np.argsort(yhat, kind="stable")
    sorted_yhat = yhat[sorted_yhat_index]
    sorted_y = y[sorted_yhat_index]

    # ignore warning for 0/0 when calculating F-measures
    prev_settings = np.seterr("ignore")

    tp = np.sum(y == 1)
    fp = y.size - tp
    fn = 0
    best_obj = micro_plus_macro(tp, fp, fn)
    best_tp, best_fp, best_fn = tp, fp, fn
    cut = -1
    pre_y_below_dec_indx, y_below_dec_indx = 0, 0
    for i in range(candidate.size):
        while (y_below_dec_indx < len(y)) and (sorted_yhat[y_below_dec_indx] <= sorted_candidate[i]):
            y_below_dec_indx += 1
        n = y_below_dec_indx - pre_y_below_dec_indx
        n_pos = sorted_y[pre_y_below_dec_indx:y_below_dec_indx].sum()
        fp -= n - n_pos
        tp -= n_pos
        fn += n_pos

        obj = micro_plus_macro(tp, fp, fn)
        if obj >= best_obj:
            best_obj = obj
            best_tp, best_fp, best_fn = tp, fp, fn
            cut = i

        pre_y_below_dec_indx = y_below_dec_indx
    np.seterr(**prev_settings)
    if cut == -1:  # i.e. all 1 in scut
        thresholds = np.nextafter(sorted_candidate[0], -np.inf)  # predict all 1
    elif cut == candidate.size - 1:
        thresholds = np.nextafter(sorted_candidate[-1], np.inf)
    else:
        thresholds = sorted_candidate[cut]#(sorted_candidate[cut] + sorted_candidate[cut + 1]) / 2

    tp_sum += best_tp
    fp_sum += best_fp
    fn_sum += best_fn

    # # In FlatModel.predict_values, the threshold is added to the decision value.
    # # Therefore, we need to make it negative here.
    # threshold = -thresholds

    threshold = thresholds
    stats["tp"] += tp_sum
    stats["fp"] += fp_sum
    stats["fn"] += fn_sum
    return threshold, stats

def dump_each_fold_thresholds(data_name, method, split, nr_fold, train_combined_folds=False):
    data_fold_thresholds = defaultdict(list)
    all_data = defaultdict(list)
  
    for fold in range(1,1+nr_fold):
    
        data = np.load(f'predict_values/{data_name}/{method}/fold_{fold}_{split}_predict_values.npz')
        logits, label = data['logits'], data['label']
        
        print(logits.shape, label.shape)

        logits_thresholds = train_thresholding(label, logits)
        simgoid_thresholds = train_thresholding(label, sigmoid(logits))
        
        data_fold_thresholds['logits'].append(logits_thresholds)
        data_fold_thresholds['sigmoid'].append(simgoid_thresholds)

        if train_combined_folds and (split == 'val'):
            all_data['logits'].append(logits)
            all_data['label'].append(label)
            all_data['sigmoid'].append(sigmoid(label))

    if split == 'test':
        data = np.load(f'predict_values/{data_name}/{method}/retrain_{split}_predict_values.npz')
        logits, label = data['logits'], data['label']

        data_fold_thresholds['retrain_logits'] = train_thresholding(label, logits)
        data_fold_thresholds['retrain_sigmoid'] = train_thresholding(label, sigmoid(logits))

    data_fold_thresholds['avg_logits'] = np.mean(data_fold_thresholds['logits'], axis=0)
    data_fold_thresholds['avg_sigmoid'] = np.mean(data_fold_thresholds['sigmoid'], axis=0)       

    if train_combined_folds:
        data_fold_thresholds['5folds_logits'] = train_thresholding_folds(all_data['label'], all_data['logits'])
        data_fold_thresholds['5folds_sigmoid'] = train_thresholding_folds(all_data['label'], all_data['sigmoid'])
                            
    return data_fold_thresholds
