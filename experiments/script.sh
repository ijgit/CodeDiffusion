ansor='../tvm-ansor/'
codediffusion='../tvm-codediffusion/'

ansor_run='ansor.py'
codediffusion_run='our.py'

model=squeezenet_v1.1
num_measures_per_round=64
num_trials=200
i=0

for target in llvm cuda; do

tag='ansor'
log_dir=log_$tag; mkdir $log_dir 
dir_name=$log_dir/$model; mkdir $dir_name
dir_name=$log_dir/$model/$i; mkdir $dir_name
(
    export TVM_HOME=$ansor 
    export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH} 
    python $ansor_run --target=$target --model=$model\
        --log_dir=$log_dir\
        --num_measures_per_round=$num_measures_per_round --test_idx=$i --num_trials=$num_trials > $dir_name/ansor-$model-$num_measures_per_round.out
)

tag='codediffusion-sketch'
log_dir=log_$tag; mkdir $log_dir 
dir_name=$log_dir/$model; mkdir $dir_name
dir_name=$log_dir/$model/$i; mkdir $dir_name
(
    export TVM_HOME=$codediffusion 
    export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH} 
    python $codediffusion_run --target=$target --model=$model\
        --log_dir=$log_dir\
        --group_type=sketch\
        --num_measures_per_round=$num_measures_per_round --test_idx=$i --num_trials=$num_trials > $dir_name/codediffusion-$model-$num_measures_per_round.out
)

tag='codediffusion-operator'
log_dir=log_$tag; mkdir $log_dir 
dir_name=$log_dir/$model; mkdir $dir_name
dir_name=$log_dir/$model/$i; mkdir $dir_name
(
    export TVM_HOME=$codediffusion 
    export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH} 
    python $codediffusion_run --target=$target --model=$model\
        --log_dir=$log_dir\
        --group_type=operator\
        --num_measures_per_round=$num_measures_per_round --test_idx=$i --num_trials=$num_trials > $dir_name/codediffusion-$model-$num_measures_per_round.out
)
done