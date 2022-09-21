# CLs
学习可持续学习的代码，用[Continvvm/continuum](https://github.com/Continvvm/continuum)，借鉴[aimagelab/mammoth](https://github.com/aimagelab/mammoth)自己造的轮子。

# run
python run_er_native.py --gpu_id 0  --buffer_size 1000  --lr 0.01 --batch_size 10 --minibatch_size 10 --n_epochs 1 --alpha 0.5 --beta 0.5 --scenario='permut_mnist'
python run_der.py --gpu_id 0  --buffer_size 1000  --lr 0.01 --batch_size 10 --minibatch_size 10 --n_epochs 1 --alpha 0.5 --beta 0.5 --scenario='permut_mnist'
