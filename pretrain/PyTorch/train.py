from datetime import datetime

import numpy as np
import random
import os
import sys
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler

import argparse
from tqdm import tqdm
from checkpoint import checkpoint_model, load_checkpoint, latest_checkpoint_file
from logger import Logger
from utils import get_sample_writer
from models import BertMultiTask
from dataset import PreTrainingDataset
from dataset import PretrainDataType
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from optimization import warmup_linear_decay_exp
from azureml_adapter import set_environment_variables_for_nccl_backend, get_local_rank, get_global_size, get_local_size
from sources import PretrainingDataCreator, TokenInstance, WikiNBookCorpusPretrainingDataCreator
from sources import WikiPretrainingDataCreator
from configuration import BertJobConfiguration

from azureml.core.run import Run


def get_effective_batch(total):
    if use_multigpu_with_single_device_per_process:
        return total//dist.get_world_size()//train_batch_size//gradient_accumulation_steps
    else:
        return total//train_batch_size//gradient_accumulation_steps # Dividing with gradient_accumulation_steps since we multiplied it earlier


def get_dataloader(dataset: Dataset, eval_set=False):
    if not use_multigpu_with_single_device_per_process:
        train_sampler = RandomSampler(dataset)
    else:
        train_sampler = DistributedSampler(dataset)
    return (x for x in DataLoader(dataset, batch_size=train_batch_size // 2 if eval_set else train_batch_size,
                                  sampler=train_sampler, num_workers=job_config.get_num_workers()))


def pretrain_validation(index):
    model.eval()
    dataset = PreTrainingDataset(tokenizer=tokenizer,
                                 folder=job_config.get_validation_folder_path(),
                                 logger=logger, max_seq_length=max_seq_length,
                                 index=index, data_type=PretrainDataType.VALIDATION,
                                 max_predictions_per_seq=max_predictions_per_seq,
                                 masked_lm_prob=masked_lm_prob)
    data_batches = get_dataloader(dataset, eval_set=True)
    eval_loss = 0
    nb_eval_steps = 0

    for batch in data_batches:
        batch = tuple(t.to(device) for t in batch)
        tmp_eval_loss = model.network(batch, log=False)
        dist.reduce(tmp_eval_loss, 0)
        # Reduce to get the loss from all the GPU's
        tmp_eval_loss = tmp_eval_loss / dist.get_world_size()
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    logger.info(f"Validation Loss for epoch {index + 1} is: {eval_loss}")
    if check_write_log():
        summary_writer.add_scalar(f'Validation/Loss', eval_loss, index + 1)
        run.log("validation_loss", np.float(eval_loss))
    return


def train(index):
    model.train()
    dataloaders = {}
    i = 0
    global global_step
    datalengths = []
    batchs_per_dataset = []

    # Pretraining datasets
    wiki_pretrain_dataset = PreTrainingDataset(tokenizer=tokenizer,
                                               folder=job_config.get_wiki_pretrain_dataset_path(),
                                               logger=logger, max_seq_length=max_seq_length,
                                               index=index, data_type=PretrainDataType.WIKIPEDIA,
                                               max_predictions_per_seq=max_predictions_per_seq,
                                               masked_lm_prob=masked_lm_prob)

    datalengths.append(len(wiki_pretrain_dataset))
    dataloaders[i] = get_dataloader(wiki_pretrain_dataset)

    num_batches_in_dataset = get_effective_batch(len(wiki_pretrain_dataset))
    logger.info('Wikpedia data file: Number of samples {}, number of batches required to process these samples: {}'.format(len(wiki_pretrain_dataset), num_batches_in_dataset))
    
    batchs_per_dataset.append(num_batches_in_dataset)
    i += 1

    logger.info("Training on Wikipedia dataset")

    if train_on_book_corpus:
        bc_pretrain_dataset = PreTrainingDataset(tokenizer=tokenizer,
                                                 folder=job_config.get_book_corpus_pretrain_dataset_path(),
                                                 logger=logger, max_seq_length=max_seq_length,
                                                 index=index, data_type=PretrainDataType.BOOK_CORPUS,
                                                 max_predictions_per_seq=max_predictions_per_seq,
                                                 masked_lm_prob=masked_lm_prob)
        datalengths.append(len(bc_pretrain_dataset))
        dataloaders[i] = get_dataloader(bc_pretrain_dataset)

        num_batches_in_dataset = get_effective_batch(len(bc_pretrain_dataset))
        batchs_per_dataset.append(num_batches_in_dataset)
        logger.info('Bookcorpus data file: Number of samples {}, number of batches required to process these samples: {}'.format(len(bc_pretrain_dataset), num_batches_in_dataset))
        
        i += 1
        logger.info("Training on Book Corpus")

    total_length = sum(datalengths)

    dataset_batches = []
    for i, batch_count in enumerate(batchs_per_dataset):
        dataset_batches.extend([i] * batch_count)
    logger.info('Number of batches to process *all* data samples in this epoch: {}'.format(len(dataset_batches)))
    # shuffle
    random.shuffle(dataset_batches)

    # We don't want the dataset to be n the form of alternate chunks if we have more than
    # one dataset type, instead we want to organize them into contiguous chunks of each
    # data type, hence the multiplication with grad_accumulation_steps with dataset_batch_type
    dataset_picker = []
    for dataset_batch_type in dataset_batches:
        dataset_picker.extend([dataset_batch_type] * gradient_accumulation_steps )

    logger.info('Number of steps to process all batches in this epoch: {}'.format(len(dataset_picker)))
    model.train()

    # Counter of sequences in an "epoch"
    sequences_counter = 0

    for step, dataset_type in enumerate(dataset_picker):
        try:
            batch = next(dataloaders[dataset_type])

            sequences_counter += len(batch)

            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)  # Move to GPU

            if step > 1 and step % 1000 == 0:
                logger.info("{} Number of sequences processed so far: {} (cumulative in {} steps)".format(datetime.utcnow(), sequences_counter, step))
            # Calculate forward pass
            loss = model.network(batch)

            if n_gpu > 1:
                # this is to average loss for multi-gpu. In DistributedDataParallel
                # setting, we get tuple of losses form all proccesses
                loss = loss.mean()

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            # Enabling  optimized Reduction
            # reduction only happens in backward if this method is called before
            # when using the distributed module
            if accumulate_gradients:
                if use_multigpu_with_single_device_per_process and (step + 1) % gradient_accumulation_steps == 0:
                    model.network.enable_need_reduction()
                else:
                    model.network.disable_need_reduction()
            if fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16:
                    # modify learning rate with special warm up BERT uses
                    # if fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = \
                        job_config.get_learning_rate() * warmup_linear_decay_exp(global_step,
                                                                                 job_config.get_decay_rate(),
                                                                                 job_config.get_decay_step(),
                                                                                 job_config.get_total_training_steps(),
                                                                                 job_config.get_warmup_proportion())
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step

                    # Record the LR against global_step on tensorboard
                    if check_write_log():
                        summary_writer.add_scalar(f'Train/lr', lr_this_step, global_step)
                    
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        except StopIteration:
            continue
            
    if check_write_log():
        run.log("training_loss", np.float(loss))
        
    logger.info("Completed {} steps".format(step))
    logger.info("Completed processing {} sequences".format(sequences_counter))

    # Run Validation Loss
    if max_seq_length == 512:
        logger.info(f"TRAIN BATCH SIZE: {train_batch_size}")
        pretrain_validation(index)


def str2bool(val):
    return val.lower() == "true" or val.lower() == "t" or val.lower() == "1"

def check_write_log():
    return dist.get_rank() == 0 or not use_multigpu_with_single_device_per_process

if __name__ == '__main__':
    print("The arguments are: " + str(sys.argv))

    parser = argparse.ArgumentParser()

    # Required_parameters
    parser.add_argument("--config_file", "--cf",
                        help="pointer to the configuration file of the experiment", type=str, required=True)

    parser.add_argument("--path", default=None, type=str, required=True,
                        help="The blob storage directory for data, config files, cache and output.")

    # Optional Params
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_predictions_per_seq", "--max_pred", default=80, type=int,
                        help="The maximum number of masked tokens in a sequence to be predicted.")
    parser.add_argument("--masked_lm_prob", "--mlm_prob", default=0.15,
                        type=float, help="The masking probability for languge model.")
    parser.add_argument("--train_batch_size", default=32,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--no_cuda",
                        type=str,
                        default='False',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--accumulate_gradients',
                        type=str,
                        default='True',
                        help="Enabling gradient accumulation optimization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        type=str,
                        default='False',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--use_pretrain',
                        type=str,
                        default='False',
                        help="Whether to use Bert Pretrain Weights or not")

    parser.add_argument('--loss_scale',
                        type=float,
                        default=0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--load_training_checkpoint', '--load_cp',
                        type=str,
                        default='False',
                        help="This is the path to the TAR file which contains model+opt state_dict() checkpointed.")

    parser.add_argument('--train_on_book_corpus',
                        type=str,
                        default='False',
                        help="Whether to train on book corpus as well")
    
    parser.add_argument('--use_multigpu_with_single_device_per_process',
                        type=str,
                        default='True',
                        help="Whether only one device is managed per process")	    

    args = parser.parse_args()

    no_cuda = str2bool(args.no_cuda)
    fp16 = str2bool(args.fp16)
    accumulate_gradients = str2bool(args.accumulate_gradients)
    use_pretrain = str2bool(args.use_pretrain)
    train_on_book_corpus = str2bool(args.train_on_book_corpus)
    use_multigpu_with_single_device_per_process = str2bool(args.use_multigpu_with_single_device_per_process)

    path= args.path
    config_file = args.config_file
    gradient_accumulation_steps = args.gradient_accumulation_steps
    train_batch_size = args.train_batch_size
    seed = args.seed
    loss_scale = args.loss_scale
    load_training_checkpoint = args.load_training_checkpoint
    max_seq_length = args.max_seq_length
    max_predictions_per_seq = args.max_predictions_per_seq
    masked_lm_prob = args.masked_lm_prob

    local_rank = -1

    local_rank = get_local_rank()
    global_size = get_global_size()
    local_size = get_local_size()	
    # TODO use logger	
    print('local_rank = {}'.format(local_rank))
    print('global_size = {}'.format(global_size))
    print('local_size = {}'.format(local_size))

    set_environment_variables_for_nccl_backend(local_size == global_size)

    # Prepare Logger
    logger = Logger(cuda=torch.cuda.is_available())

    # # Extact config file from blob storage
    job_config = BertJobConfiguration(config_file_path=os.path.join(path, config_file))
    # Replace placeholder path prefix by path corresponding to "ds.path('data/bert_data/').as_mount()"
    job_config.replace_path_placeholders(path)

    job_name = job_config.get_name()
    # Setting the distributed variables

    run = Run.get_context()

    if not use_multigpu_with_single_device_per_process:
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if fp16:
            logger.info("16-bits distributed training is not officially supported in the version of PyTorch currently used, but it works. Refer to https://github.com/pytorch/pytorch/pull/13496 for supported version.")
            fp16 = True  #
    logger.info("device: {} n_gpu: {}, use_multigpu_with_single_device_per_process: {}, 16-bits training: {}".format(
        device, n_gpu, use_multigpu_with_single_device_per_process, fp16))

    if gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            gradient_accumulation_steps))

    train_batch_size = int(train_batch_size / gradient_accumulation_steps)

    # Setting all the seeds so that the task is random but same accross processes
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # Create an outputs/ folder in the blob storage
    parent_dir = os.path.join(path, 'outputs', str(run.experiment.name))
    output_dir = os.path.join(parent_dir, str(run.id))
    os.makedirs(output_dir, exist_ok=True)
    saved_model_path = os.path.join(output_dir, "saved_models", job_name)

    summary_writer = None
    # Prepare Summary Writer and saved_models path
    if check_write_log():
        #azureml.tensorboard only streams from /logs directory, therefore hardcoded
        summary_writer = get_sample_writer(
            name=job_name, base='./logs')
        os.makedirs(saved_model_path, exist_ok=True)

    # Loading Tokenizer (vocabulary from blob storage, if exists)
    logger.info("Extracting the vocabulary")
    tokenizer = BertTokenizer.from_pretrained(job_config.get_token_file_type(), cache_dir=path)
    logger.info("Vocabulary contains {} tokens".format(len(list(tokenizer.vocab.keys()))))


    # Loading Model
    logger.info("Initializing BertMultiTask model")
    model = BertMultiTask(job_config = job_config, use_pretrain = use_pretrain, tokenizer = tokenizer, 
                          cache_dir = path, device = device, write_log = check_write_log(), 
                          summary_writer = summary_writer)

    logger.info("Converting the input parameters")
    if fp16:
        model.half()
        
    model.to(device)

    if use_multigpu_with_single_device_per_process:
        try:
            if accumulate_gradients:
                logger.info("Enabling gradient accumulation by using a forked version of DistributedDataParallel implementation available in the branch bertonazureml/apex at https://www.github.com/microsoft/apex")
                from distributed_apex import DistributedDataParallel as DDP
            else:
                logger.info("Using Default Apex DistributedDataParallel implementation")
                from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("To use distributed and fp16 training, please install apex from the branch bertonazureml/apex at https://www.github.com/microsoft/apex.")
        torch.cuda.set_device(local_rank)
        model.network = DDP(model.network, delay_allreduce=False)

    elif n_gpu > 1:
        model.network = nn.DataParallel(model.network)

    # Prepare Optimizer
    logger.info("Preparing the optimizer")
    param_optimizer = list(model.network.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    logger.info("Loading Apex and building the FusedAdam optimizer")

    if fp16:
        try:
            from apex.optimizers import FP16_Optimizer, FusedAdam
        except:
            raise ImportError("To use distributed and fp16 training, please install apex from the branch bertonazureml/apex at https://www.github.com/microsoft/apex.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=job_config.get_learning_rate(),
                              bias_correction=False,
                              max_grad_norm=1.0)
        if loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(
                optimizer, static_loss_scale=loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=job_config.get_learning_rate(),
                             warmup=job_config.get_warmup_proportion(),
                             t_total=job_config.get_total_training_steps())

    global_step = 0
    start_epoch = 0
    
    # if args.load_training_checkpoint is not None:
    if load_training_checkpoint != 'False':
        logger.info(f"Looking for previous training checkpoint.")
        latest_checkpoint_path = latest_checkpoint_file(parent_dir, no_cuda)

        logger.info(f"Restoring previous training checkpoint from {latest_checkpoint_path}")
        start_epoch, global_step = load_checkpoint(model, optimizer, latest_checkpoint_path)
        logger.info(f"The model is loaded from last checkpoint at epoch {start_epoch} when the global steps were at {global_step}")


    logger.info("Training the model")

    for index in range(start_epoch, job_config.get_total_epoch_count()):
        logger.info(f"Training epoch: {index + 1}")
        
        train(index)

        if check_write_log():
            epoch_ckp_path = os.path.join(saved_model_path, "bert_encoder_epoch_{0:04d}.pt".format(index + 1))
            logger.info(f"Saving checkpoint of the model from epoch {index + 1} at {epoch_ckp_path}")
            model.save_bert(epoch_ckp_path)
            checkpoint_model(os.path.join(saved_model_path, "training_state_checkpoint_{0:04d}.tar".format(index + 1)), model, optimizer, index, global_step)
