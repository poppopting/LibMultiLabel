CONFIG_DIR="nn_config_dropout01"
# bert
# python3 main.py --config $CONFIG_DIR/eurlex-entire/bert_eurlex_entire.yml
# python3 main.py --config $CONFIG_DIR/eurlex-entire/bert_eurlex_entire_fold_2.yml
# python3 main.py --config $CONFIG_DIR/eurlex-entire/bert_eurlex_entire_fold_3.yml
# python3 main.py --config $CONFIG_DIR/eurlex-entire/bert_eurlex_entire_fold_4.yml
# python3 main.py --config $CONFIG_DIR/eurlex-entire/bert_eurlex_entire_fold_5.yml
python3 main.py --config $CONFIG_DIR/RCV1-topics/bert_rcv1_entire.yml 
# python3 main.py --config $CONFIG_DIR/RCV1-topics/bert_rcv1_entire_fold_2.yml
# python3 main.py --config $CONFIG_DIR/RCV1-topics/bert_rcv1_entire_fold_3.yml
# python3 main.py --config $CONFIG_DIR/RCV1-topics/bert_rcv1_entire_fold_4.yml
# python3 main.py --config $CONFIG_DIR/RCV1-topics/bert_rcv1_entire_fold_5.yml
python3 main.py --config $CONFIG_DIR/hoc/bert_hoc_entire.yml
python3 main.py --config $CONFIG_DIR/go_emotions/bert_go_emotions_entire.yml
python3 main.py --config $CONFIG_DIR/pubmed/bert_pubmed_entire.yml
python3 main.py --config $CONFIG_DIR/security_ttp/bert_security_ttp_entire.yml
python3 main.py --config $CONFIG_DIR/eurlex-entire/bert_eurlex_original_entire.yml
# # bert retrain
# python3 main.py --config $CONFIG_DIR/eurlex-entire/bert_retrain_eurlex_entire.yml
# python3 main.py --config $CONFIG_DIR/RCV1-topics/bert_retrain_rcv1_entire.yml

# kim-cnn
# python3 main.py --config $CONFIG_DIR/RCV1-topics/kim_cnn_rcv1_entire.yml
# python3 main.py --config $CONFIG_DIR/eurlex-entire/kim_cnn_eurlex_entire.yml
# python3 main.py --config $CONFIG_DIR/RCV1-topics/kim_cnn_rcv1_entire_fold_2.yml
# python3 main.py --config $CONFIG_DIR/RCV1-topics/kim_cnn_rcv1_entire_fold_3.yml
# python3 main.py --config $CONFIG_DIR/RCV1-topics/kim_cnn_rcv1_entire_fold_4.yml
# python3 main.py --config $CONFIG_DIR/RCV1-topics/kim_cnn_rcv1_entire_fold_5.yml
# python3 main.py --config $CONFIG_DIR/eurlex-entire/kim_cnn_eurlex_entire_fold_2.yml
# python3 main.py --config $CONFIG_DIR/eurlex-entire/kim_cnn_eurlex_entire_fold_3.yml
# python3 main.py --config $CONFIG_DIR/eurlex-entire/kim_cnn_eurlex_entire_fold_4.yml
# python3 main.py --config $CONFIG_DIR/eurlex-entire/kim_cnn_eurlex_entire_fold_5.yml
# kim-cnn retrain
# python3 main.py --config $CONFIG_DIR/RCV1-topics/kim_cnn_retrain_rcv1_entire.yml
# python3 main.py --config $CONFIG_DIR/eurlex-entire/kim_cnn_retrain_eurlex_entire.yml

# kim-cnn only first fold
python3 main.py --config $CONFIG_DIR/hoc/kim_cnn_hoc_entire.yml
python3 main.py --config $CONFIG_DIR/go_emotions/kim_cnn_go_emotions_entire.yml
python3 main.py --config $CONFIG_DIR/pubmed/kim_cnn_pubmed_entire.yml
python3 main.py --config $CONFIG_DIR/security_ttp/kim_cnn_security_ttp_entire.yml
python3 main.py --config $CONFIG_DIR/eurlex-entire/kim_cnn_eurlex_original_entire.yml
python3 main.py --config $CONFIG_DIR/RCV1-topics/kim_cnn_rcv1_entire.yml

