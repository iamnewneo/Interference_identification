export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"
export ENV="prod"
python -u -m interf_ident.driver 2>&1 | tee out_train_log.log