Experiment 1: python main.py --seed 0 --lr 8e-5 --batch_size 1024
Experiment 2: python main.py --seed 0 --lr 8e-5 --batch_size 1024 --dim_embed 256
Experiment 3: python main.py --seed 0 --lr 8e-5 --batch_size 1024 --merging_technique padding
Experiment 4: python main.py --seed 0 --lr 8e-5 --batch_size 1024 --intermediate_emb 512
Experiment 5: python main.py --seed 0 --lr 8e-5 --batch_size 1024  # set dropout=0.2 instead of 0.5
Experiment 6: python main.py --seed 0 --lr 8e-5 --batch_size 1024  # used layernorm instead of batchnorm
