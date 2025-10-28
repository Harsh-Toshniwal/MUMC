# MUMC - Setup and Usage Instructions

## Project Overview

This is the implementation of **MUMC (Masked Vision and Language Pre-training with Unimodal and Multimodal Contrastive Losses)** for Medical Visual Question Answering (VQA). The model was accepted by MICCAI-2023.

**Paper**: [Masked Vision and Language Pre-training with Unimodal and Multimodal Contrastive Losses for Medical Visual Question Answering](https://arxiv.org/abs/2307.05314)

## System Requirements

- **OS**: Windows (PowerShell), Linux, or macOS
- **Python**: 3.11+ recommended
- **RAM**: 16GB+ recommended
- **Storage**: ~10GB for code, datasets, and checkpoints
- **GPU**: CUDA-compatible GPU highly recommended (CPU training is extremely slow)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/pengfeiliHEU/MUMC.git
cd MUMC
```

### 2. Create Virtual Environment

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```powershell
pip install torch torchvision timm transformers==4.30.0 Pillow numpy opencv-python scipy scikit-image matplotlib pyyaml
```

**Note**: The code requires `transformers==4.30.0` for compatibility with the custom BERT implementation.

## Dataset Setup

### Supported Datasets

The model supports three medical VQA datasets:
1. **VQA-RAD** (Radiology)
2. **PathVQA** (Pathology)
3. **Slake** (Medical imaging)

### VQA-RAD Dataset Setup (Example)

#### Download Dataset

1. **Official source**: https://osf.io/89kps/
2. **Alternative**: https://physionet.org/content/vqa-rad/ (requires registration)

Download the dataset and extract to a folder (e.g., `data_RAD/`).

#### Preprocess Dataset

The downloaded VQA-RAD dataset needs to be split and formatted:

```powershell
python preprocess_vqa_rad.py --input_json ofs_dataset/VQA_RAD_Dataset_Public.json --output_dir ofs_dataset
```

**Expected output structure**:
```
ofs_dataset/
├── images/                    # 315 radiology images
├── trainset.json             # 1,798 training samples
├── testset.json              # 450 test samples
└── answer_list.json          # 557 unique answers
```

#### Update Config File

Edit `configs/VQA.yaml` and update the `rad` section with your local paths:

```yaml
rad:
  train_file: ['D:/path/to/data_RAD/trainset.json']
  test_file: ['D:/path/to/data_RAD/testset.json']
  answer_list: 'D:/path/to/data_RAD/answer_list.json'
  vqa_root: 'D:/path/to/data_RAD/images/'
```

**Important**: Use forward slashes (`/`) in paths, even on Windows.

### Other Datasets

Follow similar preprocessing steps for PathVQA and Slake datasets. Refer to the main README.md for expected directory structures.

## Pretrained Weights

Download the pretrained MUMC model:

**Google Drive Link**: https://drive.google.com/file/d/1ZxwjfDeBYTMpw4mN_R9-gAcR9UOJiAFb/view?usp=sharing

- **File**: `med_pretrain_29.pth`
- **Size**: ~1.7 GB
- **Place in**: Project root directory

## Training

### Fine-tune on VQA-RAD

```powershell
python train_vqa.py --dataset_use rad --checkpoint med_pretrain_29.pth --output_dir ./output/rad --device cuda
```

**Arguments**:
- `--dataset_use`: Dataset to use (`rad`, `pathvqa`, or `slake`)
- `--checkpoint`: Path to pretrained checkpoint
- `--output_dir`: Directory to save outputs
- `--device`: `cuda` for GPU or `cpu` for CPU

### Training Configuration

Edit `configs/VQA.yaml` to adjust training parameters:

```yaml
image_res: 384              # Image resolution
batch_size_train: 8         # Training batch size
batch_size_test: 4          # Test batch size
init_lr: 2e-5              # Initial learning rate
weight_decay: 0.05          # Weight decay
max_epoch: 5               # Number of epochs
```

### Training on CPU vs GPU

**GPU (Recommended)**:
- Training time: ~1-2 hours for 5 epochs
- Requires CUDA-compatible GPU

**CPU**:
- Training time: ~15-20 hours for 5 epochs
- Very slow, not recommended for production

## Evaluation

### Evaluate Trained Model

```powershell
python train_vqa.py --dataset_use rad --checkpoint ./output/rad/checkpoint_best.pth --evaluate --device cuda
```

### Evaluate Pretrained Model

```powershell
python train_vqa.py --dataset_use rad --checkpoint med_pretrain_29.pth --evaluate --device cuda
```

**Output**:
- Accuracy metrics (overall, closed-ended, open-ended)
- Results saved to: `output_dir/result/`

## Running the Model

### Quick Start - Complete Workflow

Here's the complete workflow from setup to evaluation:

#### Step 1: Setup Environment

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Verify Python and packages
python --version
pip list | Select-String "torch|transformers|timm"
```

#### Step 2: Verify Dataset

```powershell
# Check dataset files exist
Get-ChildItem ofs_dataset\

# Expected output:
# - trainset.json
# - testset.json
# - answer_list.json
# - images\ (folder with 315 images)
```

#### Step 3: Verify Pretrained Checkpoint

```powershell
# Check checkpoint file
Get-Item med_pretrain_29.pth

# Should show ~1.7GB file
```

#### Step 4: Run Training

```powershell
# GPU training (recommended)
python train_vqa.py --dataset_use rad --checkpoint med_pretrain_29.pth --output_dir ./output/rad --device cuda

# CPU training (slower)
python train_vqa.py --dataset_use rad --checkpoint med_pretrain_29.pth --output_dir ./output/rad --device cpu
```

#### Step 5: Monitor Training Progress

```powershell
# Check training logs (in separate terminal)
Get-Content ./output/rad/log.txt -Wait

# Or check output directory for checkpoints
Get-ChildItem ./output/rad/
```

#### Step 6: Evaluate Model

```powershell
# After training completes, evaluate best checkpoint
python train_vqa.py --dataset_use rad --checkpoint ./output/rad/checkpoint_best.pth --evaluate --device cuda
```

### Running Options

#### 1. Training from Pretrained Checkpoint

**Full training (default epochs from config)**:
```powershell
python train_vqa.py --dataset_use rad --checkpoint med_pretrain_29.pth --output_dir ./output/rad --device cuda
```

**Custom number of epochs** (edit `configs/VQA.yaml` first):
```yaml
# In VQA.yaml, change:
max_epoch: 10  # Set to desired number
```

**Different datasets**:
```powershell
# PathVQA
python train_vqa.py --dataset_use pathvqa --checkpoint med_pretrain_29.pth --output_dir ./output/pathvqa --device cuda

# Slake
python train_vqa.py --dataset_use slake --checkpoint med_pretrain_29.pth --output_dir ./output/slake --device cuda
```

#### 2. Resume Training from Checkpoint

```powershell
# Resume from a saved checkpoint
python train_vqa.py --dataset_use rad --checkpoint ./output/rad/checkpoint_05.pth --output_dir ./output/rad --device cuda
```

#### 3. Evaluation Only Mode

**Evaluate without training**:
```powershell
python train_vqa.py --dataset_use rad --checkpoint med_pretrain_29.pth --evaluate --device cuda
```

**Evaluate specific checkpoint**:
```powershell
python train_vqa.py --dataset_use rad --checkpoint ./output/rad/checkpoint_best.pth --evaluate --device cuda
```

#### 4. CPU vs GPU Training

**GPU (Fast - 1-2 hours for 5 epochs)**:
```powershell
python train_vqa.py --dataset_use rad --checkpoint med_pretrain_29.pth --output_dir ./output/rad --device cuda
```

**CPU (Slow - 15-20 hours for 5 epochs)**:
```powershell
python train_vqa.py --dataset_use rad --checkpoint med_pretrain_29.pth --output_dir ./output/rad --device cpu
```

### Understanding Training Output

#### Console Output

During training, you'll see:
```
Train Epoch: [0]  [0/224]  lr: 0.000020  loss: 5.2341
Train Epoch: [0]  [10/224]  lr: 0.000020  loss: 4.8923
...
Evaluation:
overall_acc: 0.4567
closed_acc: 0.5123
open_acc: 0.4012
```

**Metrics**:
- `[0/224]`: Batch number / Total batches
- `lr`: Current learning rate
- `loss`: Training loss (should decrease)
- `overall_acc`: Accuracy on all questions
- `closed_acc`: Accuracy on closed-ended questions
- `open_acc`: Accuracy on open-ended questions

#### Output Files

After training, check `./output/rad/`:
```
output/rad/
├── checkpoint_best.pth       # Best model by validation accuracy
├── checkpoint_05.pth          # Checkpoint from epoch 5
├── checkpoint_10.pth          # Checkpoint from epoch 10
├── log.txt                    # Training log
└── result/
    ├── epoch0/
    │   ├── med_pretrain_29_vqa_result_0.json
    │   └── eval_log.txt
    ├── epoch1/
    └── ...
```

### Running Inference on New Data

To run inference on your own medical images:

#### Step 1: Prepare Data

Create a JSON file with your questions:
```json
[
  {
    "image_name": "image001.jpg",
    "question": "What organ is shown in this image?",
    "answer": "",
    "qid": 1
  },
  {
    "image_name": "image002.jpg", 
    "question": "Is there an abnormality?",
    "answer": "",
    "qid": 2
  }
]
```

#### Step 2: Update Config

Edit `configs/VQA.yaml`:
```yaml
rad:
  test_file: ['path/to/your/questions.json']
  vqa_root: 'path/to/your/images/'
  answer_list: 'ofs_dataset/answer_list.json'  # Use existing answer list
```

#### Step 3: Run Inference

```powershell
python train_vqa.py --dataset_use rad --checkpoint ./output/rad/checkpoint_best.pth --evaluate --device cuda
```

Results will be saved to `output_dir/result/`.

### Monitoring Training

#### Method 1: Watch Log File

```powershell
# Real-time log monitoring
Get-Content ./output/rad/log.txt -Wait -Tail 20
```

#### Method 2: Check Checkpoints

```powershell
# List saved checkpoints
Get-ChildItem ./output/rad/*.pth | Select-Object Name, Length, LastWriteTime
```

#### Method 3: Parse Results

```powershell
# Check latest evaluation results
Get-Content ./output/rad/result/epoch*/eval_log.txt | Select-String "acc"
```

### Troubleshooting Running Issues

#### Training Doesn't Start

1. **Check GPU availability**:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

2. **Check dataset paths** in `configs/VQA.yaml`

3. **Verify checkpoint exists**:
   ```powershell
   Test-Path med_pretrain_29.pth
   ```

#### Training Crashes

1. **Out of Memory**: Reduce batch size in `configs/VQA.yaml`
2. **CUDA Error**: Switch to CPU with `--device cpu`
3. **File Not Found**: Check all paths use forward slashes

#### Poor Performance

1. **Low accuracy**: Train for more epochs (increase `max_epoch`)
2. **Overfitting**: Reduce learning rate or add regularization
3. **Underfitting**: Increase learning rate or train longer

### Performance Expectations

#### VQA-RAD Benchmark

With pretrained model (`med_pretrain_29.pth`):

| Metric | Expected Accuracy |
|--------|------------------|
| Overall | 62-67% |
| Closed-ended | 75-82% |
| Open-ended | 55-60% |

After 40 epochs of fine-tuning:
- Overall: ~70%
- Closed-ended: ~85%
- Open-ended: ~62%

**Note**: Results may vary based on hyperparameters and random seed.

## Project Structure

```
MUMC/
├── configs/
│   ├── VQA.yaml              # VQA configuration
│   ├── Pretrain.yaml         # Pretraining configuration
│   └── config_bert.json      # BERT configuration
├── dataset/
│   ├── vqa_dataset.py        # VQA dataset loader
│   ├── pretrain_dataset.py   # Pretraining dataset
│   └── utils.py              # Dataset utilities
├── models/
│   ├── model_vqa.py          # MUMC VQA model
│   ├── model_pretrain.py     # Pretraining model
│   ├── xbert.py              # Custom BERT implementation
│   └── vision/               # Vision encoder (ViT)
├── vqaTools/                 # VQA evaluation tools
├── train_vqa.py              # Main training script
├── pretrain.py               # Pretraining script
├── preprocess_vqa_rad.py     # Dataset preprocessing
└── requirements.txt          # Python dependencies
```

## Common Issues and Solutions

### 1. Import Errors

**Error**: `ModuleNotFoundError: No module named 'ruamel_yaml'`

**Solution**: Install PyYAML instead:
```powershell
pip install pyyaml
```

### 2. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size in `configs/VQA.yaml`
- Use gradient accumulation
- Use CPU (slower but works)

### 3. Checkpoint Loading Issues

**Error**: `RuntimeError: size mismatch for visual_encoder_m.pos_embed`

**Solution**: The code automatically handles position embedding resizing. Ensure you're using the updated `train_vqa.py`.

### 4. Long Training Time on CPU

**Issue**: Training takes 15-20 hours on CPU

**Solutions**:
- Use GPU (reduces to 1-2 hours)
- Reduce number of epochs
- Use smaller batch size

### 5. Path Issues on Windows

**Issue**: Paths with backslashes not working

**Solution**: Always use forward slashes in config files:
```yaml
# Correct
vqa_root: 'D:/USC/Deep Learning/MUMC/ofs_dataset/images/'

# Incorrect
vqa_root: 'D:\USC\Deep Learning\MUMC\ofs_dataset\images\'
```

## Dataset Statistics

### VQA-RAD
- **Total samples**: 2,248 QA pairs
- **Training**: 1,798 samples (80%)
- **Test**: 450 samples (20%)
- **Images**: 315 radiology images
- **Unique answers**: 557
- **Question types**: Presence, Position, Abnormality, Modality, etc.

### Dataset Distribution
- **Closed-ended questions**: ~46% (Yes/No, single-choice)
- **Open-ended questions**: ~54% (free-form answers)

## Performance Metrics

The model is evaluated on:
1. **Overall Accuracy**: Accuracy across all questions
2. **Closed-ended Accuracy**: Accuracy on closed questions
3. **Open-ended Accuracy**: Accuracy on open questions

## GPU Setup (Recommended)

### Install CUDA PyTorch

```powershell
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify GPU

```python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Your GPU name
```

## Additional Resources

- **Paper**: https://arxiv.org/abs/2307.05314
- **MICCAI 2023**: https://conferences.miccai.org/2023
- **ALBEF (Inspiration)**: https://github.com/salesforce/ALBEF

