#! /bin/sh

#SBATCH --job-name=train_resnet
#SBATCH --output=/home/yandex/DL20222023a/gottesman3/RESNET_OCD/train_output/train_resnet.out # redirect stdout
#SBATCH --error=/home/yandex/DL20222023a/gottesman3/RESNET_OCD/train_output/train_resnet.err # redirect stderr
#SBATCH --partition=studentbatch # (see resources section)
#SBATCH --time=720 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=100000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=2 # GPUs in total

python run_func_OCD.py -e 0 -pb ./base_models/resnet20.pt -pc ./configs/train_resnet.json -pdtr ./data/cifar10 -pdts ./data/cifar10 -dt resnet -prc 0
