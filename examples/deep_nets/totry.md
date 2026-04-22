--decay-factor 0.35
--decay-factor 0.65
RUNNING --n-buckets 10 --decay-factor 0.5
--no-feature_weight_standardization
--classifier_norm none
--no-feature-weight-standardization --classifier-norm none
--feature-norm layernorm
--no-feature_weight_standardization --feature-norm layernorm


BEST T=80 (200 epochs, suggest run with scheduler first...): run_dvsgesture.py --architecture vgg9_mid --optimizer adamw --weight-decay 5e-4 --lr 5e-4 --scheduler cosine   --scheduler-eta-min 1e-5 
--device cuda:1

