import os 
import pickle
from micromacro import dump_each_fold_thresholds
from micromacro_utils import dump_predict_value




if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    datasets = ['eurlex-entire', 'RCV1-topics']#['eurlex-entire', 'RCV1-topics']
    methods = ['bert'] #['bert', 'kim-cnn'] 

    nr_fold = 5
    working_dir = '/home/ktdev/nn_micromacro/nn_micromacro'
    os.chdir(working_dir)
    runs_base_dir = './runs' 
    predict_save_folder = './dropout02_predict_values'#'./predict_values'

    #download prediction value
    for data_name in datasets:
        for method in methods:
            # each fold
            logger.info('Begin fetching the prediction data and save them into folder')
            logger.info('Begin proceed each fold')
            for fold in range(1,1+nr_fold):
                dump_predict_value(method, data_name, data_type=fold,
                                splits=['val', 'test'],
                                model_name='best_model.ckpt',
                                runs_base_dir=runs_base_dir,
                                save_folder=predict_save_folder)
                
            logger.info('Begin proceed retrain version')
            # retrain
            dump_predict_value(method, data_name, data_type='retrain',
                            splits=['test'],
                            model_name='last.ckpt',
                            runs_base_dir=runs_base_dir,
                            save_folder=predict_save_folder)

            logger.info('Data have been dumped')

    logger.info('load data from .npz file and do thresholding')
    for data_name in datasets:
        for method in methods:
            data_fold_thresholds = {
                'val': dump_each_fold_thresholds(data_name, method, 'val', nr_fold, train_combined_folds=True),
                'test': dump_each_fold_thresholds(data_name, method, 'test', nr_fold, train_combined_folds=False)
            }
            pickle.dump(data_fold_thresholds, open(f'{predict_save_folder}/{data_name}/{method}/data_fold_thresholds.pkl', 'wb'))
