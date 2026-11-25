# Shared defaults for the b70-prod build and serve scripts (sourced, not run). Every value can be overridden
# from the environment. Image tags match scripts/build-images-034.sh.
export INKLING2_RUNTIME=${INKLING2_RUNTIME:-local/b70-inkling-vllm:034-runtime}
export INKLING2_BUILDER=${INKLING2_BUILDER:-local/inkling2-oneapi-034:2026.1.1}
export INKLING2_CXX_STD=${INKLING2_CXX_STD:-c++20}
# The upstream scripts stop at 22:44 Europe/Berlin unless a deadline is given; this profile has no daily cut-off.
export INKLING2_DEADLINE_EPOCH=${INKLING2_DEADLINE_EPOCH:-4102444800}
