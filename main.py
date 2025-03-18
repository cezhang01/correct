import argparse
import random
import datetime
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from transformers import BertConfig
import numpy as np
import pickle
from data_loader import *
from model import Model
from evaluation import *
import os
import time
from tqdm import tqdm


def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters for model
    parser.add_argument('-dn', '--dataset_name', type=str, default='check_covid')
    parser.add_argument('-m', '--mode', type=str, default='supervised', choices=['supervised', 'few_shot', 'test'])
    parser.add_argument('-ns', '--num_shots', type=int, default=5, help='used only when mode == few_shot')
    parser.add_argument('-es', '--evidence_setting', type=str, default='gold', choices=['gold', 'retrieved'])
    parser.add_argument('-np', '--num_prompt_embs', type=int, default=8)  # for supervised version, use 8 or more, for few-shot or zero-shot, use 4
    parser.add_argument('-rp', '--random_prompt_init', type=bool, default=False)
    parser.add_argument('-lm', '--language_model', type=str, default='pubmed_bert', choices=['base_bert', 'sci_bert', 'pubmed_bert', 'large_bert', 'multilingual_bert'])
    parser.add_argument('-ne', '--num_epochs', type=int, default=50)
    parser.add_argument('-ls', '--log_steps', type=int, default=5)
    parser.add_argument('-lr_lm', '--learning_rate_for_lm', type=float, default=1e-5)
    parser.add_argument('-lr_p', '--learning_rate_for_prompt_embs', type=float, default=1e-3)
    parser.add_argument('-ms', '--minibatch_size', type=int, default=2)
    parser.add_argument('-ml', '--max_text_length', type=int, default=256)
    parser.add_argument('-n_evid', '--num_sampled_evidence', type=int, default=3)
    parser.add_argument('-n_ref', '--num_sampled_references', type=int, default=5)

    # hyperparameters for training
    parser.add_argument('-ddp', '--distributed_training', type=bool, default=False)
    parser.add_argument('-gpu', '--gpu', type=int, default=0, help='used only when ddp is False')
    parser.add_argument('-rs', '--random_seed', type=int, default=519)

    return parser.parse_args()


def set_random_seed(random_seed):

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True


def cleanup():

    dist.destroy_process_group()


def load_data(args):

    if args.language_model == 'base_bert':
        args.pretrained_model_name_or_path = 'bert-base-uncased'
    elif args.language_model == 'sci_bert':
        args.pretrained_model_name_or_path = 'allenai/scibert_scivocab_uncased'
    elif args.language_model == 'pubmed_bert':
        args.pretrained_model_name_or_path = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
    elif args.language_model == 'large_bert':
        args.pretrained_model_name_or_path = 'bert-large-cased'
    elif args.language_model == 'multilingual_bert':
        args.pretrained_model_name_or_path = 'bert-base-multilingual-cased'

    data_center = DataCenter(args)
    training_data = Data(data_center, mode=args.mode, num_shots=args.num_shots)
    test_data = Data(data_center, mode='test', num_shots=args.num_shots)

    training_sampler = DistributedSampler(training_data, shuffle=True) if args.distributed_training else RandomSampler(training_data)
    test_sampler = SequentialSampler(test_data)

    if args.mode == 'few_shot' and data_center.num_labels * args.num_shots < args.minibatch_size:
        args.minibatch_size = data_center.num_labels * args.num_shots

    training_loader = DataLoader(training_data, batch_size=args.minibatch_size, sampler=training_sampler)
    test_loader = DataLoader(test_data, batch_size=1, sampler=test_sampler)

    return data_center, training_loader, test_loader, training_data


def train(args):

    # Setup CUDA, GPU & distributed training
    if not args.distributed_training:
        args.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        args.local_rank = -1
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=36000))
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.global_rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank)
        torch.cuda.empty_cache()
        args.device = torch.device('cuda:' + str(args.local_rank))

    set_random_seed(args.random_seed + args.local_rank)

    if args.local_rank in [-1, 0]:
        print('******************************************************')
        print('********************** training **********************')
        print('******************************************************')

    if args.local_rank in [-1, 0]:
        print('Loading data...')
    data_center, training_loader, test_loader, training_data = load_data(args)

    if args.local_rank in [-1, 0]:
        print('Loading model...')
    model = Model(args, data_center).to(args.device)

    if args.local_rank in [-1, 0]:
        print(model)

    # define DDP here
    if args.distributed_training:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    else:
        ddp_model = model

    if args.local_rank in [-1, 0]:
        print('Start training...')

    optimizer = torch.optim.Adam([
        {'params': [param for name, param in ddp_model.named_parameters() if 'prompt_learner' not in name], 'lr': args.learning_rate_for_lm},
        {'params': [param for name, param in ddp_model.named_parameters() if 'prompt_learner' in name], 'lr': args.learning_rate_for_prompt_embs}],
        lr=args.learning_rate_for_lm)

    t = time.time()
    for epoch_id in range(1, args.num_epochs + 1):
        # training
        one_epoch_loss = 0.0
        ddp_model.train()
        if args.distributed_training:
            training_loader.sampler.set_epoch(epoch_id)
        data_center.sample_evid_and_ref()
        for batch_id, batch in tqdm(enumerate(training_loader), total=len(training_loader)):
            claim_ids, labels = batch
            optimizer.zero_grad()
            res = ddp_model(claim_ids, labels, data_center, mode=args.mode)
            loss = res[0]
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                one_epoch_loss += loss.item()
            if args.distributed_training:
                torch.distributed.barrier()
        one_epoch_loss /= len(training_loader)
        # validation
        if epoch_id % args.log_steps == 0 and args.local_rank in [-1, 0]:
            print('******************************************************')
            print('Time: %ds' % (time.time() - t), '\tEpoch: %d/%d' % (epoch_id, args.num_epochs), '\tLoss: %f' % one_epoch_loss)
            ckpt_folder_exists = os.path.exists('./ckpt')
            if not ckpt_folder_exists:
                os.makedirs('./ckpt')
            mode = str(args.num_shots) + '_shot' if args.mode == 'few_shot' else args.mode
            setting = 'gold_evidence' if args.evidence_setting == 'gold' else 'retrieved_evidence'
            torch.save(model.state_dict(), './ckpt/' + args.dataset_name + '_' + setting + '_' + mode + '_' + args.language_model + '.pt')
            test(model, data_center, test_loader)
        if args.distributed_training:
            torch.distributed.barrier()

    if args.distributed_training:
        cleanup()


def test(model, data_center, test_loader):

    model.eval()
    y_true, y_pred = [], []
    for batch_id, batch in enumerate(test_loader):
        claim_ids, labels = batch
        res = model(claim_ids, labels, data_center, mode='test')
        y_pred.extend(res[1].detach().cpu().numpy().tolist())
        y_true.extend(labels.detach().cpu().numpy().tolist())
    y_pred = np.array(y_pred)[:data_center.num_test_claims]
    y_true = np.array(y_true)[:data_center.num_test_claims]
    classification(y_pred, y_true)


def main(args):

    if args.mode != 'test':
        train(args)
    else:
        ################## You should use single GPU for testing. ####################
        print('******************************************************')
        print('********************** testing ***********************')
        print('******************************************************')
        args.distributed_training = False
        args.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        args.local_rank = -1
        set_random_seed(args.random_seed)
        data_center, training_loader, test_loader, training_data = load_data(args)
        model = Model(args, data_center).to(args.device)
        mode = str(args.num_shots) + '_shot' if args.mode == 'few_shot' else args.mode
        setting = 'gold_evidence' if args.evidence_setting == 'gold' else 'retrieved_evidence'
        ckpt = torch.load('./ckpt/' + args.dataset_name + '_' + setting + '_' + mode + '_' + args.language_model + '.pt', map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
        test(model, data_center, test_loader)


if __name__ == '__main__':
    main(parse_args())