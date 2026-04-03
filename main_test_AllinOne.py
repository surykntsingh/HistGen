import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import numpy as np
from modules.tokenizers import Tokenizer, MedicalReportTokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.tester_AllinOne import Tester
from modules.loss import compute_loss
from models.histgen_model import HistGenModel
import os
import random
def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='/lustre/nvwulf/projects/PrasannaGroup-nvwulf/surya/datasets/REG_2025/train/20x_512px_0px_overlap/features_conch_v15', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='/lustre/nvwulf/projects/PrasannaGroup-nvwulf/surya/datasets/REG_2025/reg_reports_splits.json', help='the path to the directory containing the data.')
    # parser.add_argument('--image_dir', type=str, default='/lustre/nvwulf/projects/PrasannaGroup-nvwulf/surya/datasets/TCGA_processed/brca_v2/features_conch_v15', help='the path to the directory containing the data.')
    # parser.add_argument('--ann_path', type=str, default='/lustre/nvwulf/projects/PrasannaGroup-nvwulf/surya/datasets/TCGA_processed/brca_v2/tcga_brca_reports_splits.json', help='the path to the directory containing the data.')


    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='TCGA', choices=['TCGA','HistAI','REG'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=600, help='the maximum sequence length of the reports.')
    parser.add_argument('--max_fea_length', type=int, default=10000, help='the maximum sequence length of the patch embeddings.')

    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=1, help='the number of samples for a batch')
    
    parser.add_argument('--model_name', type=str, default='histgen', choices=['histgen'], help='model used for experiment')

    # Model settings (for visual extractor)
    # parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    # parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')
    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    # for Cross-modal context module
    parser.add_argument('--topk', type=int, default=32, help='the number of k.')
    parser.add_argument('--cmm_size', type=int, default=2048, help='the numebr of cmm size.')
    parser.add_argument('--cmm_dim', type=int, default=512, help='the dimension of cmm dimension.')
    # for Local-global hierachical visual encoder
    parser.add_argument("--region_size", type=int, default=256, help="the size of the region for region transformer.")
    parser.add_argument("--prototype_num", type=int, default=512, help="the number of visual prototypes for cross-modal interaction")

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=str, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/REG/3/', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/REG/3/', help='the patch to save the results of experiments')
    parser.add_argument('--load', type=str, default='results/REG/3/', help='whether to load a pre-trained model.')

    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')
    parser.add_argument('--log_period', type=int, default=1000, help='the logging interval (in batches).')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    # parser.add_argument('--load', type=str, help='whether to load a pre-trained model.')

    args = parser.parse_args()
    return args


def setup(gpus):
    # Let torchrun set these; fallback for safety/debug
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # Force loopback
    os.environ['MASTER_PORT'] = '30001'  # Or any free port >1024master_port
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    dist.init_process_group(backend='nccl')

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def main():
    # parse arguments
    args = parse_agrs()
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # scaling learning rate
    args.lr_ed *= world_size

    setup(args.n_gpu)
    torch.cuda.set_device(local_rank)
    init_seeds(args.seed + local_rank)
    
    # tokenizer = Tokenizer(args)
    tokenizer = MedicalReportTokenizer(args)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
    model = HistGenModel(args, tokenizer).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build trainer and start to train
    tester = Tester(model, criterion, metrics, args, test_dataloader)
    tester.test()


if __name__ == '__main__':
    main()
