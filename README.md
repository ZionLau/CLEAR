# CLEAR

## Running

### 1) Install Dependencies
We provide a `requirements.txt` for environment setup:

```bash
pip install -r requirements.txt
```

### 2) Training

#### Weibo19
```bash
python train.py
```

#### VRDD
```bash
python train_VRDD.py
```

### 3) Evaluation (VRDD)

#### Test a single checkpoint (default: runs both TEST and OOD/VAL)
```bash
python test.py Lambdas_MaxVALAcc_0.9311.bin
```

#### Test multiple checkpoints (batch comparison)
```bash
python test.py A.bin B.bin C.bin
```

#### OOD only
```bash
python test.py Lambdas_MaxVALAcc_0.9311.bin --run ood
```

## Data Availability
All datasets and processed files will be released after the paper is accepted.
