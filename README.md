# Robotic Prediction Comparison

Benchmarks three trajectory prediction models (SSM, LSTM, TCN) on the ETH/UCY pedestrian dataset, then feeds the predictions into a potential-field navigation controller to simulate a robot avoiding pedestrians.

## Core Files

| File | What it does |
|------|--------------|
| `preprocess.py` | Converts raw ETH/UCY `.txt` files into `(N, 20, 2)` PyTorch tensors |
| `ssm/model.py` | S4D-based trajectory predictor |
| `ssm/train.py` | Trains the SSM and saves the best checkpoint |
| `ssm/test.py` | Evaluates the SSM, prints ADE/FDE, saves prediction plots |
| `lstm/model.py` | LSTM-based trajectory predictor |
| `lstm/train.py` | Trains the LSTM |
| `lstm/test.py` | Evaluates the LSTM |
| `tcn/tcn_model.py` | TCN-based trajectory predictor |
| `tcn/tcn_train.py` | Trains the TCN |
| `tcn/tcn_test.py` | Evaluates the TCN |
| `navigation/controller.py` | Potential-field controller (attractive + repulsive forces) |
| `navigation/simulate_simple.py` | Runs the ego robot through the Univ scene using any of the three models |
| `navigation/animate.py` | Renders a saved rollout as an `.mp4` or `.gif` |

## Data

Raw pedestrian tracks live in `datasets/{scene}/`. Processed tensors are written to `datasets_processed/{scene}/{train,val,test}.pt`. Each tensor has shape `(N, 20, 2)` -- 8 observed frames followed by 12 prediction frames, agent-centric (last observed position at origin).

## Group Contributions

Hallie - TCN implementation
Adelaide - LSTM implementation
Jacob - Data preprocessing, SSM implementation, robototic controller rollout