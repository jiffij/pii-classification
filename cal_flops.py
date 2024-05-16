import torch
from thop import profile
from thop import clever_format
from model import SHARNN
from transformers import MobileBertForTokenClassification, MobileBertConfig

# Define your model

# MODEL = "checkpoints\\BERT.pt"
# MODEL = "ep3_sha_cuda_1head_logweight_Odiv10_10k_fastem_hisattn.pt"
# MODEL = "2_cont_pretrained_cos.pt"
# MODEL = "sha_cuda_4head_logweight_10k.pt"
# MODEL = "pii_cuda_nomsk.pt"

bert = False
use_fast = False

def model_load(fn):
    global model#, criterion, optimizer
    with open(fn, 'rb') as f:
        #torch.nn.Module.dump_patches = True
        #model, criterion, optimizer = torch.load(f)
        #model, criterion = torch.load(f)
        # m, criterion = torch.load(f)
        m, _, _, _, _ = torch.load(f) #criterion, optimizer, lr_scheduler, warmup_scheduler
        d = m.state_dict()
        #del d['pos_emb']
        model.load_state_dict(d, strict=False)
        print("pretrained model loaded...")
        del m

if not bert:
    model = SHARNN("LSTM", 1024 if use_fast else 128000, 1024, 4096, 4)
else:
    cfg = MobileBertConfig(vocab_size=128000, num_labels=13, max_position_embeddings=1024)
    model = MobileBertForTokenClassification(cfg)

model_load(MODEL)

with torch.no_grad():
    


    # Create a dummy input tensor
    if not use_fast:
        input = torch.ones(1024, 1, dtype=torch.long)
    else:
        input = torch.randn(1024, 1024 if use_fast else 1)

    # Use the profile function to calculate FLOPs and parameters
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}, Params: {params}")