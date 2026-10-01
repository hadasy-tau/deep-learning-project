# The persistent volume has a ~10 GB quota (probed 2026-09-20). Model weights and the
# xet chunk cache go on the 100 GB container disk (re-downloadable in minutes); the
# token stays under HF_HOME on the volume; audio, results and adapters stay in the repo.
export HF_HUB_CACHE=/root/hfcache/hub
export HF_XET_CACHE=/root/hfcache/xet
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
