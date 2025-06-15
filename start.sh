nohup python train_text.py --config_path configs/cnn.yaml > nohup_logs/nohup_cnn_img.log 2>&1 &
nohup python train.py --config_path configs/cnn.yaml > nohup_logs/nohup_cnn_multi.log 2>&1 &
nohup python evaluation.py > nohup_logs/eval.log 2>&1 &

nohup python train.py --config_path configs/multimodal_cnn_D1.yaml > nohup_logs/multimodal_cnn_D1.log 2>&1 &
nohup python train.py --config_path configs/multimodal_lstm_D1.yaml > nohup_logs/multimodal_lstm_D1.log 2>&1 &
nohup python train.py --config_path configs/multimodal_bert_D1.yaml > nohup_logs/multimodal_bert_D1.log 2>&1 &

nohup python train.py --config_path configs/multimodal_cnn_recovery.yaml > nohup_logs/multimodal_cnn_recovery.log 2>&1 &
nohup python train.py --config_path configs/multimodal_lstm_recovery.yaml > nohup_logs/multimodal_lstm_recovery.log 2>&1 &
nohup python train.py --config_path configs/multimodal_bert_recovery.yaml > nohup_logs/multimodal_bert_recovery.log 2>&1 &


nohup python train.py --config_path configs/unimodal_text_cnn_D1.yaml > nohup_logs/unimodal_text_cnn_D1.log 2>&1 &
nohup python train.py --config_path configs/unimodal_text_lstm_D1.yaml > nohup_logs/unimodal_text_lstm_D1.log 2>&1 &
nohup python train.py --config_path configs/unimodal_text_bert_D1.yaml > nohup_logs/unimodal_text_bert_D1.log 2>&1 &
nohup python train.py --config_path configs/unimodal_img_vgg_D1.yaml > nohup_logs/unimodal_img_vgg_D1.log 2>&1 &

nohup python train.py --config_path configs/unimodal_text_cnn_recovery.yaml > nohup_logs/unimodal_text_cnn_recovery.log 2>&1 &
nohup python train.py --config_path configs/unimodal_text_lstm_recovery.yaml > nohup_logs/unimodal_text_lstm_recovery.log 2>&1 &
nohup python train.py --config_path configs/unimodal_text_bert_recovery.yaml > nohup_logs/unimodal_text_bert_recovery.log 2>&1 &
nohup python train.py --config_path configs/unimodal_img_vgg_recovery.yaml > nohup_logs/unimodal_img_vgg_recovery.log 2>&1 &