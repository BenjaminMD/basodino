# env.sh: source before any IGM run on s03 (bash). Keeps caches inside the project.
IGM_HOME=$HOME/dev/AGL/basodino/igm
export PATH=$IGM_HOME/env/bin:$HOME/.local/bin:$PATH
export TMPDIR=$IGM_HOME/tmp                  # pip, XLA and nrun temporaries (not /tmp)
export KERAS_HOME=$IGM_HOME/cache/keras      # Keras would otherwise create ~/.keras
export MPLCONFIGDIR=$IGM_HOME/cache/mpl
export CUDA_CACHE_PATH=$IGM_HOME/cache/nv    # PTX -> sm_90 JIT cache (TF 2.17 ships compute_90 PTX)
export CUDA_CACHE_MAXSIZE=4294967296
export TF_FORCE_GPU_ALLOW_GROWTH=true        # igm_run sets memory growth itself; this covers plain python
export TF_CPP_MIN_LOG_LEVEL=2
# choose the GPU per run: CUDA_VISIBLE_DEVICES=0 or 1
