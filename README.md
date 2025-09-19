# ml_utils
A ml_utils repository containing the following sections:
- [Trainer](#trainer-)



## Trainer 🌟
Trainer class to provide support for training any machine learning model. The goal is to make it reusable to save time for setting up the boilerplate code. Some functions are basic for training any model and then there are some advanced functions. 
### Basic functions
- Training loop
- Evaluation loop
- checkpointing
- logging training and evaluation metrics
- Customized optimizer
- Customized learning rate schedular
### Advanced
- Multi-GPU support
- Gradient accumulation

## Usage Example 💡
The repository provides a **base trainer class** (`BaseTrainer`) available via:

    from ml_utils import BaseTrainer

You can extend this class to implement customized training workflows.  
Examples of such customizations are provided in [ml_utils/trainers/trainer.py](./ml_utils/trainers/trainers.py).


## Updated Support 🆕
### Current Features (v0.1)

### Planned Improvements

## Change Log 📜