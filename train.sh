export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"
export ENV="prod"
python -m interf_ident.driver | tee out_train_log.log