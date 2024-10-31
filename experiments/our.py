import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor
import argparse
import os

import transformers
from transformers import *
import torch
import pickle
from transformers import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import copy


def task_desc_to_group_name(
    task_desc
):
    task_layers = task_desc.split('_')
    if task_layers[-1].isdigit():
        task_layers.pop()
    return '_'.join(task_layers)

def args_parser():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--target', help='target hardware')
    parser.add_argument('--model', help='model name')
    parser.add_argument('--num_measures_per_round', type=int, default=64, help='number of measure trials')
    parser.add_argument('--num_trials', type=int, default=200, help='number of measure trials per task')
    parser.add_argument('--test_idx', type=int, default=0, help='test idx')
    parser.add_argument('--group_name', default=None)
    parser.add_argument('--log_dir', default='log')
    parser.add_argument('--group_type', default='operator')
    args = parser.parse_args()
    return args

# Define the neural network and compilation target
args = args_parser()
# if args.target != 'llvm':
#     # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     # os.environ["CUDA_VISIBLE_DEVICES"] = args.target
#     device = 'cuda'
# else:
#     device = 'llvm'

device = args.target
log_dir = args.log_dir
dir_name=f"{log_dir}/{args.model}/{args.test_idx}"

network = args.model
batch_size = 1
if network == 'squeezenet_v1.1' or network == 'mxnet':
    layout = "NCHW"
else:
    layout = "NHWC"
batch_size = 1
target = tvm.target.Target(device) # cpu: llvm
dtype = "float32"
# rm_ratio = args.rm
test_idx = args.test_idx
num_measures_per_round = args.num_measures_per_round
num_trials = args.num_trials
group_name = args.group_name
group_type = args.group_type
# num_target_tune = args.num_target_tune
# target_task_idx = args.target_task_idx

def get_network(name, batch_size, layout="NHWC", dtype="float32"):
    """Get the symbol definition and random weight of a network"""

    from mxnet.gluon.model_zoo.vision import get_model
    from gluoncv2.model_provider import get_model as glcv2_get_model

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "mobilenetv2":
        block = get_model("mobilenetv2_1.0", pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": (batch_size, 3, 224, 224)}, dtype=dtype
        )
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
        if layout == "NHWC":
            desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
            seq = tvm.transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
        
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
        
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
        
    elif name == "efficientnet":

        block = net = glcv2_get_model("EfficientNet_B0", pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": (batch_size, 3, 224, 224)}, dtype=dtype
        )
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
        if layout == "NHWC":
            desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
            seq = tvm.transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
        
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
        
    elif "densenet" in name:
        mod, params = relay.testing.densenet.get_workload(batch_size=batch_size, dtype=dtype)
        if layout == "NHWC":
            desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
            seq = tvm.transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
        
    elif name == "bert":
        from transformers import BertTokenizer
        from transformers import BertConfig
        from transformers import BertModel

        enc = BertTokenizer.from_pretrained("bert-base-uncased")
        # Tokenizing input text
        text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
        tokenized_text = enc.tokenize(text)

        # Masking one of the input tokens
        masked_index = 8
        tokenized_text[masked_index] = "[MASK]"
        indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

        # Creating a dummy input
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        dummy_input = [tokens_tensor, segments_tensors]

        # Initializing the model with the torchscript flag
        # Flag set to True even though it is not necessary as this model does not have an LM Head.
        config = BertConfig(
            vocab_size_or_config_json_file=32000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            torchscript=True,
        )
        # Instantiating the model
        model = BertModel(config)
        # The model needs to be in evaluation mode
        model.eval()
        # If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
        model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)
        # Creating the trace
        traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])

        shape_list = [
            (i.debugName().split(".")[0], i.type().sizes())
            for i in list(traced_model.graph.inputs())[1:]
        ]

        mod, params = tvm.relay.frontend.pytorch.from_pytorch(
            traced_model, shape_list, default_dtype="float32"
        )
        input_shape = tokens_tensor.numpy()
        output_shape = segments_tensors.numpy()
        
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.vgg.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            dtype=dtype,
        )
        if layout == "NHWC":
            desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
            seq = tvm.transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)

    return mod, params, input_shape, output_shape

# Extract tasks from the network
print("Extract tasks...")
mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

# test
# for task in tasks:
#     print(task.desc)

# added for amd cpu
task_weights = [weight for task, weight in zip(tasks, task_weights) if 'vm_mod_fused_nn_dense_add' not in task.desc]
tasks = [task for task in tasks if 'vm_mod_fused_nn_dense_add' not in task.desc]

# tune all tasks

if not os.path.exists(f"./{dir_name}/"):
    os.makedirs(f"./{dir_name}")

log_file = f"./{dir_name}/our-{network}-{layout}-B{batch_size}-{target.kind.name}-{num_measures_per_round}-{group_type}.json"
log_file_name = f"./{dir_name}/our-{network}-{layout}-B{batch_size}-{target.kind.name}-{num_measures_per_round}-{group_type}.tsv"

# grouped_tasks = {}; grouped_weights = {}
# for task, weight in zip(tasks, task_weights):
#     group_name = task_desc_to_group_name(task.desc)
#     if group_name not in grouped_tasks:
#         grouped_tasks[group_name] = []
#         grouped_weights[group_name] = []
#     grouped_tasks[group_name].append(task)
#     grouped_weights[group_name].append(weight)

# # resort tasks in group by cosine similarity
# for group_name in grouped_tasks.keys():
#     input_shapes = {}
#     for i, task in enumerate(grouped_tasks[group_name]):
#         input_shapes[i] = eval(task.workload_key)[1:]
#         input_shapes[i] = np.array(input_shapes[i]).flatten()
#     input_shapes = pd.DataFrame(input_shapes)
        
#     cosine_sim = cosine_similarity(input_shapes.T)
#     cosine_sim_df = pd.DataFrame(cosine_sim, index=input_shapes.columns, columns=input_shapes.columns)
#     mean_cosine_sim = cosine_sim_df.mean(axis=0)

#     max_index = mean_cosine_sim.idxmax()

#     # sort tasks in group by cosine similarity
#     sorted_tasks = [grouped_tasks[group_name][max_index]]
#     sorted_task_weights = [grouped_weights[group_name][max_index]]
#     for i in range(len(grouped_tasks[group_name])):
#         if i == max_index:
#             pass
#         else:
#             sorted_tasks.append(grouped_tasks[group_name][i])
#             sorted_task_weights.append(grouped_weights[group_name][i])
            
#     grouped_tasks[group_name] = sorted_tasks
#     grouped_weights[group_name] = sorted_task_weights
    

# for group_name in grouped_tasks.keys():
#     print(f"Group: {group_name}")
#     for task in grouped_tasks[group_name]:
#         print(task.desc , task.compute_dag)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    task.id = idx
    
def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights, log_file_name=log_file_name, group_type=group_type)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials = len(tasks)*num_trials, # * 800, # 2000, #800*6, # len(tasks) * 800 #200,  # change this to 20000 to achieve the best performance
        early_stopping = None,
        num_measures_per_round = num_measures_per_round,
        verbose=1,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    tuner.tune(tune_option)
    
print("Begin tuning... all tasks")
start_time = time.time()
run_tuning()
end_time = time.time()
execution_time = end_time - start_time
print(f"all tasks Execution time: {execution_time} seconds")


# log_file = f"./{dir_name}/ansor-{network}-{test_group_name}-{layout}-B{batch_size}-{target.kind.name}-{num_measures_per_round}-all-nt{num_target_tune}-target_min.json"
# log_file_name = f"./{dir_name}/ansor-{network}-{test_group_name}-{layout}-B{batch_size}-{target.kind.name}-{num_measures_per_round}-all-nt{num_target_tune}-target_min.tsv"
    
# def run_tuning():
#     print("Begin tuning...")
#     measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

#     tuner = auto_scheduler.TaskScheduler(tasks_min_first, task_min_weights, log_file_name=log_file_name)
#     tune_option = auto_scheduler.TuningOptions(
#         num_measure_trials = len(tasks_min_first)*num_trials, # * 800, # 2000, #800*6, # len(tasks) * 800 #200,  # change this to 20000 to achieve the best performance
#         early_stopping = None,
#         num_measures_per_round = num_measures_per_round,
#         verbose=1,
#         runner=measure_ctx.runner,
#         measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
#     )
#     tuner.tune(tune_option, num_target_tune=num_target_tune, target_task_idx=0)

# print("Begin tuning... all tasks")
# start_time = time.time()
# run_tuning()
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"all tasks Execution time: {execution_time} seconds")


# # tune single task
# for target_task_idx in range(len(tasks)):

#     if not os.path.exists(f"./{dir_name}/{target_task_idx}"):
#         os.makedirs(f"./{dir_name}/{target_task_idx}")

#     log_file = f"./{dir_name}/{target_task_idx}/ansor-{network}-{test_group_name}-{layout}-B{batch_size}-{target.kind.name}-{num_measures_per_round}-t{target_task_idx}-nt{num_target_tune}.json"
#     log_file_name = f"./{dir_name}/{target_task_idx}/ansor-{network}-{test_group_name}-{layout}-B{batch_size}-{target.kind.name}-{num_measures_per_round}-t{target_task_idx}-nt{num_target_tune}.tsv"


#     # rm task without target_task_idx
#     task_weights = [weight for idx, (task, weight) in enumerate(zip(tasks, task_weights)) if idx == target_task_idx]
#     _tasks = [task for idx, task in enumerate(tasks) if idx == target_task_idx]

#     for idx, task in enumerate(_tasks):
#         print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
#         print(task.compute_dag)
        
#     def run_tuning():
#         print("Begin tuning...")
#         measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

#         tuner = auto_scheduler.TaskScheduler(_tasks, task_weights, log_file_name=log_file_name)
#         tune_option = auto_scheduler.TuningOptions(
#             num_measure_trials = len(_tasks)*num_trials, # * 800, # 2000, #800*6, # len(tasks) * 800 #200,  # change this to 20000 to achieve the best performance
#             early_stopping = None,
#             num_measures_per_round = num_measures_per_round,
#             verbose=1,
#             runner=measure_ctx.runner,
#             measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
#         )
#         print("[0] target_task_idx: ", target_task_idx)
#         tuner.tune(tune_option, target_task_idx=target_task_idx, num_target_tune=num_target_tune)
#         # with open('./result.pkl', 'wb') as f:
#         #     pickle.dump(tuner.results, f)
        
#     print("Begin tuning... target_task_idx: ", target_task_idx)
#     start_time = time.time()
#     run_tuning()
#     end_time = time.time()
#     execution_time = end_time - start_time
#     print(f"task {target_task_idx} Execution time: {execution_time} seconds")
