### 1. Create virtualenv (recommended)

```
conda create -n harpo python=3.10
conda activate harpo
```

### 2. Install packages

```
pip install -r requirements.txt
```

### 3. Create Api config file

Create `api_setting.json` file in `HARPO/configs` directory and insert the following contents (GPT-4o is recommended):

```
{
    "endpoints": "<base_url>/chat/completions",
    "api_key": "sk-xxx",
    "model": "xxx"
}
```

## Experiments

### 1. HumanEval

#### 1.1 Download HumanEval datasets

```
pip install human-eval
```

#### 1.2 Run script

```
python ./experiment/MMLU/run.py
```

### 2. MATH

#### 2.1 Download MATH datasets

Link: https://huggingface.co/datasets/qwedsacf/competition_math

#### 2.2 Unzip datasets and rename `MATH_data`, move it into `HARPO/experiment/MATH` directory

#### 2.3 Run script

```
python ./experiment/MATH/run.py
```

### 3. MMLU

#### 3.1 Download MMLU datasets

Link: https://people.eecs.berkeley.edu/~hendrycks/data.tar

#### 3.2 Unzip datasets and rename `MMLU_data`, move it into `HARPO/experiment/MMLU` directory

#### 3.3 Run script

```
python ./experiment/MMLU/run.py
```

