mkdir -p logs

# python train.py --run baseline > logs/log_baseline_small_lr.txt

# python train.py --run geom_01 > logs/log_geom_01.txt
# python train.py --run geom_02 > logs/log_geom_02.txt
# python train.py --run geom_03 > logs/log_geom_03.txt

# python train.py --run pool > logs/log_pool.txt

python train.py --run mutual_01 > logs/log_mutual_01.txt
python train.py --run mutual_02 > logs/log_mutual_02.txt
python train.py --run mutual_03 > logs/log_mutual_03.txt

python train.py --run desc_01 > logs/log_desc_01.txt
python train.py --run desc_02 > logs/log_desc_02.txt
python train.py --run desc_03 > logs/log_desc_03.txt
