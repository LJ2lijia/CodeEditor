#!/usr/bin/env python
import os
import argparse


def get_cmd(task, sub_task, model_tag, gpu, zero_shot, data_num, bs, lr, source_length, target_length, patience, epoch, warmup,
            model_dir, summary_dir, res_fn, max_steps=None, save_steps=None, log_steps=None):
    # if max_steps is None:
    cmd_str = 'bash exp_with_args.sh %s %s %s %s %s %d %d %d %d %d %d %d %d %s %s %s' % \
                (task, sub_task, model_tag, gpu, zero_shot, data_num, bs, lr, source_length, target_length, patience, epoch,
                warmup, model_dir, summary_dir, res_fn)
    # else:
    #     cmd_str = 'bash exp_with_args.sh %s %s %s %s %d %d %d %d %d %d %d %d %s %s %s %d %d %d' % \
    #               (task, sub_task, model_tag, gpu, data_num, bs, lr, source_length, target_length, patience, epoch,
    #                warmup, model_dir, summary_dir, res_fn, max_steps, save_steps, log_steps)
    return cmd_str


def get_args_by_task_model(task, sub_task):
    if task == 'repair':
        # small: Read 46680 examples, avg src len: 31, avg trg len: 28, max src len: 50, max trg len: 50
        # [TOKENIZE] avg src len: 50, avg trg len: 45, max src len: 129, max trg len: 121
        # medium:  Read 52364 examples, avg src len: 74, avg trg len: 73, max src len: 100, max trg len: 100
        # [TOKENIZE] avg src len: 117, avg trg len: 114, max src len: 238, max trg len: 238
        if sub_task == 'small':
            src_len = 150
            trg_len = 150
        elif sub_task == 'medium':
            src_len = 300
            trg_len = 300
        epoch = 50
        patience = 5
    elif task == 'review':
        # small: Read 46680 examples, avg src len: 31, avg trg len: 28, max src len: 50, max trg len: 50
        # [TOKENIZE] avg src len: 50, avg trg len: 45, max src len: 129, max trg len: 121
        # medium:  Read 52364 examples, avg src len: 74, avg trg len: 73, max src len: 100, max trg len: 100
        # [TOKENIZE] avg src len: 117, avg trg len: 114, max src len: 238, max trg len: 238
        if sub_task == 'tufano':
            src_len = 256
            trg_len = 256
            epoch = 100
            patience = 10
        elif sub_task == 'large':
            src_len = 400
            trg_len = 400
            epoch = 20
            patience = 3
    elif task == 'concode':
        src_len = 320
        trg_len = 150
        epoch = 30
        patience = 3

    bs = 32
    if sub_task == 'tufano':
        bs = 16
    lr = 5

    return bs, lr, src_len, trg_len, patience, epoch


def run_one_exp(args):
    bs, lr, src_len, trg_len, patience, epoch = get_args_by_task_model(args.task, args.sub_task)
    print('============================Start Running==========================')
    cmd_str = get_cmd(task=args.task, sub_task=args.sub_task, model_tag=args.model_tag, gpu=args.gpu, zero_shot=args.zero_shot,
                      data_num=args.data_num, bs=bs, lr=lr, source_length=src_len, target_length=trg_len,
                      patience=patience, epoch=epoch, warmup=1000,
                      model_dir=args.model_dir, summary_dir=args.summary_dir,
                      res_fn='{}/{}_{}.txt'.format(args.res_dir, args.task, args.model_tag))
    print('%s\n' % cmd_str)
    os.system(cmd_str)


def run_multi_task_exp(args):
    # Total train data num = 1149722 (for all five tasks)
    if 'codet5_small' in args.model_tag:
        bs, lr, max_steps, save_steps, log_steps = 60, 5, 600000, 20000, 100
    else:
        bs, lr, max_steps, save_steps, log_steps = 25, 5, 800000, 20000, 100

    if args.data_num != -1:
        max_steps, save_steps, log_steps = 1000, 200, 50
    print('============================Start Running==========================')
    cmd_str = get_cmd(task='multi_task', sub_task='none', model_tag=args.model_tag, gpu=args.gpu,
                      data_num=args.data_num, bs=bs, lr=lr, source_length=-1, target_length=-1,
                      patience=-1, epoch=-1, warmup=1000,
                      model_dir=args.model_dir, summary_dir=args.summary_dir,
                      res_fn='{}/multi_task_{}.txt'.format(args.res_dir, args.model_tag),
                      max_steps=max_steps, save_steps=save_steps, log_steps=log_steps)
    print('%s\n' % cmd_str)
    os.system(cmd_str)


def get_sub_tasks(task):
    if task == 'repair':
        sub_tasks = ['small', 'medium']
    elif task == 'review':
        sub_tasks = ['tufano', 'large']
    elif task == 'concode':
        sub_tasks = ['default']
    else:
        raise ValueError('Invalid task: {}'.format(task))
    return sub_tasks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", type=str, default='editor_small',
                        choices=['roberta', 'codebert', 'graphcodebert', 'codet5_small', 'codet5_base', 'editor_small'], required=True)
    parser.add_argument("--task", type=str, default='repair', choices=['repair', 'review', 'concode'], required=True)
    parser.add_argument("--sub_task", type=str, default='small', required=True)
    parser.add_argument("--zero_shot", type=str, default='False', choices=['True', 'False'])
    parser.add_argument("--res_dir", type=str, default='results', help='directory to save fine-tuning results')
    parser.add_argument("--model_dir", type=str, default='saved_models', help='directory to save fine-tuned models')
    parser.add_argument("--summary_dir", type=str, default='tensorboard', help='directory to save tensorboard summary')
    parser.add_argument("--data_num", type=int, default=-1, help='number of data instances to use, -1 for full data')
    parser.add_argument("--gpu", type=str, default=0, help='index of the gpu to use in a cluster')
    args = parser.parse_args()

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    assert args.sub_task in get_sub_tasks(args.task)
    if args.task != 'multi_task':
        run_one_exp(args)
    else:
        run_multi_task_exp(args)
