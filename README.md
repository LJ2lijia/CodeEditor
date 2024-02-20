# CodeEditor

Offical implementation of our paper "CodeEditor: Learning to Edit Source Code with Pre-trained Models" in TOSEM. [Paper](https://lj2lijia.github.io/papers/CodeEditor_TOSEM.pdf)

## Dependency
- Python 3.8
- pytorch 1.7.1
- transformers 4.14.1
- tensorboard 2.4.1
- tqdm 4.62.1
- nltk 3.6.7
- numpy 1.19.2
- datasets 1.12.1

## Pre-trained Model

We release the pre-trained checkpoint of CodeEditor (60M) on [Google Drive](https://drive.google.com/file/d/1zTqdXq9Z_KVg1ZP4-0m2vMOA_EKQZr9q/view?usp=sharing). Please download and put it in the root directory (i.e., `CodeEditor/editor-small`).

## Dataset

In our paper, we fine-tune the pre-trained CodeEditor on four datasets, including `repair_small`, `repair_medium`, `review_tufano`, and `review_large`. You can download the datasets from [Google Drive](https://drive.google.com/file/d/1ahAReJ4gnewoqFicamd6a7Vy6nugNSNd/view?usp=sharing). Please put the datasets in the root directory (i.e., `CodeEditor/data`).

## Fine-tuning

Before the fine-tuning, please get your working path to CodeEditor and add it to the first line of `sh/exp_with_args.sh`.
For example:
```bash
WORKDIR="/home/allen/CodeEditor"
```

To fine-tune the pre-trained CodeEditor on a specific dataset, you can run the following command:

```bash
cd sh
python run_exp.py --model_tag editor_small --task {task} --sub_task {sub_task} --gpu {gpt_id}
```
where `{task}` is the task name (`repair` or  `review`), `{sub_task}` is a specific dataset under the task (`repair`: `small` or `medium`; `review`: `tufano` or `large`). `{gpt_id}` is the GPU id (e.g., `0`).

## Zero-Shot Evaluation

You can evaluate the zero-shot performance of the pre-trained CodeEditor on the four datasets by running the following command:

```bash
cd sh
python run_exp.py --model_tag editor_small --task {task} --sub_task {sub_task} --gpu {gpt_id} --zero_shot True
```

## Citation
If you find our work helpful, please consider citing our paper:
```
@article{CodeEditor,
  author       = {Jia Li and
                  Ge Li and
                  Zhuo Li and
                  Zhi Jin and
                  Xing Hu and
                  Kechi Zhang and
                  Zhiyi Fu},
  title        = {CodeEditor: Learning to Edit Source Code with Pre-trained Models},
  journal      = {{ACM} Trans. Softw. Eng. Methodol.},
  volume       = {32},
  number       = {6},
  pages        = {143:1--143:22},
  year         = {2023},
  url          = {https://doi.org/10.1145/3597207},
  doi          = {10.1145/3597207},
  timestamp    = {Thu, 09 Nov 2023 21:13:34 +0100},
  biburl       = {https://dblp.org/rec/journals/tosem/LiLLJHZF23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Acknowledgement
We thank the authors of the following repositories for their great work: [CodeT5](https://github.com/salesforce/CodeT5).