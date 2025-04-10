import os
import pickle
import datetime
import numpy as np
import libmultilabel.linear as linear
from micromacro_after_nn_train.micromacro import train_thresholding

def load_datapath():
    datapath = {}
    for dataset in ['eurlex_lexglue', 'go_emotions', 'hoc', 'pubmed', 'rcv1_topics', 'security_ttp']:
        datafolder = os.path.join('./data', dataset)
        if dataset == 'rcv1_topics':
            train = './data/rcv1_topics/rcv1_topics_train_fold_1.txt'
            val = './data/rcv1_topics/rcv1_topics_val_fold_1.txt'
            test = './data/rcv1_topics/rcv1_topics_test.txt'
        else:
            train = os.path.join(datafolder, f'{dataset}_raw_texts_train.txt')
            val = os.path.join(datafolder, f'{dataset}_raw_texts_val.txt')
            test = os.path.join(datafolder, f'{dataset}_raw_texts_test.txt')

        datapath[dataset] = {
            'train': train,
            'val': val,
            'test': test,
        }
        assert os.path.exists(train)
        assert os.path.exists(val)
        assert os.path.exists(test)
        
    return datapath


def run_one_fold_and_store_vals(datapath, dataname):
    
    datasets = linear.load_dataset("txt", datapath[dataname]['train'], datapath[dataname]['val'])
    preprocessor = linear.Preprocessor()
    datasets = preprocessor.fit_transform(datasets)
    test_datasets = linear.load_dataset("txt", datapath[dataname]['test'])
    test_datasets = preprocessor.transform(test_datasets)

    try: 
        # model = linear.train_thresholding(y=datasets["train"]["y"], x=datasets["train"]["x"], multiclass=False) # '-s 0'
        for solver in ['BCE', 'hingeSQ']:
            options = '-s 0' if solver == 'BCE' else '-s 1'
            model = linear.train_1vsrest(y=datasets["train"]["y"], x=datasets["train"]["x"], multiclass=False, options=options) 

            predict_save_folder = os.path.join('linear_runs', f'{solver}-linear-{dataname}')
            os.makedirs(predict_save_folder, exist_ok=True)

            val_preds = model.predict_values(datasets["test"]["x"])
            test_preds = model.predict_values(test_datasets["train"]["x"])

            now = str(datetime.datetime.now().strftime("%m%d-%H%M"))
            file_name = 'linear_val_predict_values.npz'
            np.savez_compressed(os.path.join(predict_save_folder, file_name), logits=val_preds, label=datasets["test"]["y"].toarray())

            now = str(datetime.datetime.now().strftime("%m%d-%H%M"))
            file_name = 'linear_test_predict_values.npz'
            np.savez_compressed(os.path.join(predict_save_folder, file_name), logits=test_preds, label=test_datasets["train"]["y"].toarray())

            # do micromacro
            logits_thresholds = train_thresholding(datasets["test"]["y"].toarray(), val_preds)
            data_fold_thresholds = {
                'val': {'logits': logits_thresholds}
            }
            pickle.dump(data_fold_thresholds, open(f'{predict_save_folder}/data_fold_thresholds.pkl', 'wb'))
    
    except Exception as e:
        print(e)
        return 'fail'

    return 'success'


if __name__ == '__main__':
    
    datapath = load_datapath()
    for dataname in ['go_emotions', 'eurlex_lexglue', 'hoc', 'pubmed', 'rcv1_topics', 'security_ttp']:
        status = run_one_fold_and_store_vals(datapath, dataname)
        assert status == 'success'
