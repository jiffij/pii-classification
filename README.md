# Project Description

This project used SHA-RNN and MobileBERT to classify Personal Identity Information (PII). The dataset is collected from Kaggle (The Learning Agency Lab - PII Data Detection)[https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data]. 

|Training Setting                              | SHA-RNN | MobileBERT |
|-----------------------------------|----------|------------|
| learning rate | 1 × 10−2 | 1 × 10−2 |
| Loss function | Log-weighted Cross-Entropy | Weighted Cross-Entropy |
| Optimizer | AdamW | AdamW |
| Dropout | 0.3 | 0.3 |
| Warmup | 1 epoch linear | 1 epoch linear |
|lr scheduler | CosineAnnealing with WarmRestarts | CosineAnnealing with WarmRestarts |
| Attention layers | 4 | 3 |
| Attention Head | 1 | 4 |
| Embedding size | 1024 | 128 |

|Result                              | SHA-RNN | MobileBERT |
|-----------------------------------|----------|------------|
| F5 | 0.7| 0.98 |
| Precision | 0 | 0.84 |
| Recall | 1 | 0.99
| Tag Accuracy | 0.438 | 0.917 |
| Parameters | 51.485M | 20.350M |
| FLOPs | 51.704G | 20.743G |
| Average runtime (s/epoch) | 3238.6 | 719.73 |
| Device | RTX 4070 | RTX 4090 |

MobileBERT shows excellent results both in accuracy metrics and computational efficiency. With a nearly perfect recall, meaning that the model has the ability to find all the relevant cases within a dataset, a low complexity due to its low number of parameters, and good computational efficiency with the lowest average time per epoch of all implemented models, MobileBERT showcases as an excellent alternative to tackle the ongoing necessity of protecting users PII.

## Changed Items 
- CosineAnnealingWarmRestarts
- LinearWarmup
- Bi-directional LSTM
- Removed attention mask
- Changed to Pytorch multi-headed attn
- Optimizer -> adam, adamw
- changed SplitCEloss to CEloss (with weight, or log weight) --ce_weight
- removed amp
- added pii dataset
- changed to multi-headed attn.


# Single Headed Attention RNN

For full details see the paper [Single Headed Attention RNN: Stop Thinking With Your Head](https://arxiv.org/abs/1911.11423).

In summary, "stop thinking with your (attention) head".

- Obtain strong results on a byte level language modeling dataset (enwik8) in under 24 hours on a single GPU (12GB Titan V)
- Support long range dependencies (up to 5000 tokens) without increasing compute time or memory usage substantially by using a simpler attention mechanism
- Avoid the fragile training process required by standard Transformer models such as a long warmup
- Back off toward a standard LSTM allowing you to drop retained memory states (needed for a Transformer model) if memory becomes a major constraint
- Provide a smaller model that features only standard components such as the LSTM, single headed attention, and feed-forward modules such that they can easily be productionized using existing optimized tools and exported to various formats (i.e. ONNX)

| Model                             | Test BPC | Params | LSTM Based |
|-----------------------------------|----------|--------|------------|
| Krause mLSTM                      | 1.24     | 46M    | ✔          |
| AWD-LSTM                          | 1.23    | 44M    | ✔          |
| **SHA-LSTM**                          | 1.07     | 63M    | ✔          |
| 12L Transformer-XL                | 1.06     | 41M    |            |
| 18L Transformer-XL                | 1.03     | 88M    |            |
| Adaptive Span Transformer (Small) | 1.02     | 38M    |            |

Whilst the model is still quite some way away from state of the art (~0.98 bpc) the model is low resource and high efficiency without having yet been optimized to be so.
The model was trained in under 24 hours on a single GPU with the [Adaptive Span Transformer](https://github.com/facebookresearch/adaptive-span) (small) being the only recent Transformer model to achieve similar levels of training efficiency.

## To recreate

### Setup

To get started:

- Retrieve the data with `./getdata.sh`
- Install PyTorch version 1.2+
- Install Nvidia's [AMP](https://github.com/NVIDIA/apex)
- Install the minimum trust variant of LAMB from [Smerity's PyTorch-LAMB](https://github.com/Smerity/pytorch-lamb)

Run the following code to install pytorch-lamb:

`cd pytorch-lamb &
pip install -e .`

### Training the model

By default the model trains the minimal single headed attention model from the paper, inserting a lone attention mechanism in the second last layer of a four layer LSTM.
This takes only half an hour per epoch on a Titan V or V100.
If you want slightly better results but a longer training time (an hour per epoch) set `use_attn` to True for all layers in `model.py` and decrease batch size until it fits in memory (i.e. 8).
Sadly there are no command line options for running the other models - it's manual tinkering.
The code is not kind.
I'll be performing a re-write in the near future meant for long term academic and industrial use - contact me if you're interested :)

Note: still [shaking out bugs from the commands below](https://github.com/Smerity/sha-rnn/issues/3). We have near third party replication but still a fix or two out. Feel free to run and note any discrepancies! If you fiddle with hyper-parameters (which I've done very little of - it's a treasure chest of opportunity to get a lower than expected BPC as your reward!) do report that too :)

When running the training command below continue until the validation bpc stops improving. Don't worry about letting it run longer as the code will only save the model with the best validation bpc.

`python -u main.py --epochs 32 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --data data/enwik8/ --save ENWIK8.pt --log-interval 10 --seed 5512 --optimizer lamb --bptt 1024 --warmup 800 --lr 2e-3 --emsize 1024 --nhid 4096 --nlayers 4 --batch_size 16`

When the training slows down a second pass with a halved learning rate until validation bpc stops improving will get a few more bpc off. A smart learning rate decay is likely the correct way to go here but that's not what I did for my experiments.

`python -u main.py --epochs 5 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --data data/enwik8/ --save ENWIK8.pt --log-interval 10 --seed 5512 --optimizer lamb --bptt 1024 --warmup 800 --lr 2e-3 --emsize 1024 --nhid 4096 --nlayers 4 --batch_size 16 --resume ENWIK8.pt --lr 1e-3 --seed 125`

Most of the improvement will happen in the first few epochs of this final command.

The final test bpc should be approximately 1.07 for the full 4 layer SHA-LSTM or 1.08 for the single headed 4 layer SHA-LSTM.


## Reference
MobileBERT: https://arxiv.org/abs/2004.02984
MobileBERT HuggingFace: https://huggingface.co/docs/transformers/model_doc/mobilebert
SHA-RNN: https://arxiv.org/abs/1911.11423
SHA-RNN git repo: https://github.com/Smerity/sha-rnn

