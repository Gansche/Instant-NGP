./scripts/delete.sh
CUDA_VISIBLE_DEVICES=2 python instant_ngp.py --data_path chair
CUDA_VISIBLE_DEVICES=2 python instant_ngp.py --data_path drums
CUDA_VISIBLE_DEVICES=2 python instant_ngp.py --data_path ficus
CUDA_VISIBLE_DEVICES=2 python instant_ngp.py --data_path hotdog