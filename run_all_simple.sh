mkdir -p logs

# python train.py --run att_baseline > logs/att_baseline.txt
# python train.py --run geom_01 --layers 1 > logs/log_geom_01_l1.txt
python train.py --run geom_01 --layers 3 > logs/log_geom_01_l3.txt

# python train.py --run desc_only --layers 2 > logs/log_desc_only_2layers.txt
# python train.py --run graph_only --layers 2 > logs/log_graph_only_2layers.txt

# python train.py --run baseline > logs/log_baseline_small_lr.txt

# python train.py --run geom_01 > logs/log_geom_01.txt
# python train.py --run geom_02 > logs/log_geom_02.txt
# python train.py --run geom_03 > logs/log_geom_03.txt

# python train.py --run pool > logs/log_pool.txt

# python train.py --run mutual_01 > logs/log_mutual_01.txt
# python train.py --run mutual_02 > logs/log_mutual_02.txt
# python train.py --run mutual_03 > logs/log_mutual_03.txt

# python train.py --run desc_01 > logs/log_desc_01.txt
# python train.py --run desc_02 > logs/log_desc_02.txt
# python train.py --run desc_03 > logs/log_desc_03.txt
