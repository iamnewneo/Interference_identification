export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"
python -m interf_ident.driver | tee train_log.log