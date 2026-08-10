#!/bin/bash
#add 2026-08-07
source /opt/dtk/env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_DEBUG=INFO
export HIPBLASLT_ALLOW_TF32=1
export HIP_VISIBLE_DEVICES=0,1,2,3

> ${LOG}
echo ">>> [1/5] tests/L0/run_rocm.sh" >> ${LOG}
cd ${WORK_SPACE}/tests/L0 && bash ./run_rocm.sh >> ${LOG} 2>&1
echo "${SEP}" >> ${LOG}
echo ">>> [2/5] tests/distributed/run_rocm_distributed.sh" >> ${LOG}
cd ${WORK_SPACE}/tests/distributed && sh ./run_rocm_distributed.sh >> ${LOG} 2>&1
echo "${SEP}" >> ${LOG}
echo ">>> [3/5] tests/L0/run_optimizers/test_fused_optimizer.py" >> ${LOG}
cd ${WORK_SPACE}/tests/L0/run_optimizers/ && python test_fused_optimizer.py >> ${LOG} 2>&1
echo "${SEP}" >> ${LOG}
echo ">>> [4/5] apex/contrib/test/run_rocm_extensions.py" >> ${LOG}
cd ${WORK_SPACE}/apex/contrib/test/  && python run_rocm_extensions.py >> ${LOG} 2>&1
echo "${SEP}" >> ${LOG}
echo ">>> [5/5] apex/contrib/test/test_label_smoothing.py" >> ${LOG}
cd ${WORK_SPACE}/apex/contrib/test/  && python test_label_smoothing.py >> ${LOG} 2>&1
echo "apex test完成"

