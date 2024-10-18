# EvoPress

Code for ICLR 2025 submission `EvoPress: Towards Optimal Dynamic Model Compression via Evolutionary Search`.
 
## Usage

### Repository structure
---

- ```src/``` — root directory with repository source code \
    ```├── evo_drop.py``` — evolutionary depth pruning \
    ```├── greedy_drop.py``` — greedy depth pruning \
    ```├── brute_drop.py``` — brute depth pruning \
    ```├── prune.py``` — SparseGPT unstructured pruning (preparation of database for EvoPress) \
    ```├── owl_prune.py``` — SparseGPT unstructured pruning (preparation of database for OWL) \
    ```├── quant.py``` — GPTQ quantization (preparation of database for EvoPress) \
    ```├── lmeval.py``` — LM Eval Harness evalution script \
    ```├── eval_ppl.py``` — perplexity evalution script

### Calibration data
---

We provide 3 options for calibration data: `wikitext2`, `c4`, `fineweb_edu`.
We recommend using the latter one for the best results. In our experiments we used **8M** tokens
for calibration. To prepare a specific amount of calibration data specify
`--calibration_samples`, `--calibration_sequence_length` for their product to be equal to desired number of tokens, i.e `--calibration_samples=1024`, `--calibration_sequence_length=8192` for 8M tokens.

We used `--calibration_samples=1024` and `--calibration_sequence_length=8192` for Llama-3-8B, Phi-3-medium-128k-instruct, and  `--calibration_samples=2048` and `--calibration_sequence_length=4096` for Llama-2-7b.

### Multi-GPU
---

Some of the scripts (Unstructured Sparsity, Quantization) may operate in **distributed** mode
for faster execution. We recommend using `torchrun` to launch them:

```shell
torchrun --nnodes=1 --nproc-per-node=<NUM_GPU> <name_of_the_script.py> <args...>
```

### Depth pruning
---

We provide 3 scripts for Depth pruning:
* `evo_drop.py` — evolutionary depth pruning
* `greedy_drop.py` — greedy depth pruning
* `brute_drop.py` — evolutionary depth pruning

### Unstructured Sparsity
---

We provide 2 scripts for Depth pruning:
* `prune.py` —  SparseGPT unstructured pruning (preparation of database for EvoPress)
* `owl_prune.py` — SparseGPT unstructured pruning (preparation of database for OWL)

### Quantization
---

We provide `quant.py` producing GPTQ database for EvoPress.

### Evaluation
---

We provide `lmeval.py` and `eval_ppl.py` scripts for evaluation on [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) benchmarks and perplexity measurements. The interface of  `lmeval.py` mostly follows the instructions from the original. In addition, one should specify the path to sparse/quantized weights via `--sparse-weights-path`/`--quant-weights-path` argument and path to `.txt` with chosen compression levels via `--sparse-config-path`/`--quant-config-path` argument. We adopted `lm-eval==0.4.0` for evaluation. 

## Environment

This code was tested on the following environment:
```
pytorch                   2.4.0           py3.10_cuda12.1_cudnn9.1.0_0    pytorch
pytorch-cuda              12.1                 ha16c6d3_5    pytorch
cuda                      12.1.0                        0    nvidia/label/cuda-12.1.0
transformers              4.43.4                   pypi_0    pypi
datasets                  2.21.0                   pypi_0    pypi
lm-eval                   0.4.0                    pypi_0    pypi
```

## Notes

Scripts `prune.py`, `owl_prune.py`, `quant.py` produce several versions of compressed representation
for each weight `(100-200 Gb)`. Make sure that you have sufficient amount of free space on drive before running.
