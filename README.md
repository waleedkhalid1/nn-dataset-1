# Neural Network Dataset
The original version of this dataset was created by <strong>Arash Torabi Goodarzi, Roman Kochnev</strong> and <strong>Zofia Antonina Bentyn</strong> at the Computer Vision Laboratory, University of Würzburg, Germany.

## Installation of the Latest Version of the NN-Dataset

```bash
pip install git+https://github.com/ABrain-One/nn-dataset.git
```

## Environment for NN-Dataset Developers
### Pip package manager
Create a virtual environment, activate it, and run the following command to install all the project dependencies:
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

### Docker
All versions of this project are compatible with <a href='https://hub.docker.com/r/abrainone/ai-linux' target='_blank'>AI Linux</a> and can be run inside a Docker image:
```bash
docker run -v /a/mm:<nn-dataset path> abrainone/ai-linux bash -c "PYTHONPATH=/a/mm python ab/nn/train.py"
```

## Usage

The primary goal of NN-Dataset project is to provide flexibility for dynamically combining various datasets, metrics, and models. It is designed to facilitate the validation of neural network performance across different training hyperparameter combinations and data transformation algorithms, while also generating performance statistics. It is primarily developed to support the <a href="https://github.com/ABrain-One/nn-gen"> NN-Gen</a> project.

The main usage scenario:
1. Add a new neural network model into the `ab/nn/datasets` directory.
2. Create a new configuration folder for its training, e.g., `ab/nn/stat/img_classification-cifar10-acc-cifar10_norm-ComplexNet`
3. Run the automated training process for the new model:
```bash
python ab/nn/train.py
```

## Contribution

To add more neural network models to the dataset, the following criteria must be met:
1. The code for each model must be provided in a respective ".py" file for the model in the directory <strong>/ab/nn/dataset</strong>. This file must be named after the name of the model structure.
2. The main class for each model must be named <strong>Net</strong>.
3. The implementation of this <strong>Net</strong> class must provide non-mutable default parameters for its constructor.
4. For each pull request involving a new neural network, please generate and submit training statistics for 100 Optuna trials (or at least 3 trials for very large models) in the <strong>ab/nn/stat</strong> directory. The trials should cover 1, 2, and 5 epochs of training. Ensure that this statistics is included along with the model in your pull request. For example, the statistics for the ComplexNet model are stored in three separate folders, each containing two files - <strong>trials.json</strong> and <strong>best_trials.json</strong>:<br/>
img_classification-cifar10-acc-cifar10_norm-ComplexNet/1<br/>
img_classification-cifar10-acc-cifar10_norm-ComplexNet/2<br/>
img_classification-cifar10-acc-cifar10_norm-ComplexNet/5<br/>


For examples, see the models in the <strong>/ab/nn/dataset</strong> directory and statistics in the <strong>ab/nn/stat</strong> directory.

### Available Modules

The `nn-dataset` package includes the following key modules:

1. **Datasets**:
   - Predefined neural network architectures such as `AlexNet`, `ResNet`, `VGG`, and more.
   - Located in `ab.nn.dataset`.

2. **Loaders**:
   - Data loaders for datasets such as CIFAR-10 and COCO.
   - Located in `ab.nn.loader`.

3. **Metrics**:
   - Common evaluation metrics like accuracy and IoU.
   - Located in `ab.nn.metric`.

4. **Utilities**:
   - Helper functions for training and statistical analysis.
   - Located in `ab.nn.util`.


## Citation

If you find Neural Network Dataset to be useful for your research, please consider citing:
```bibtex
@misc{ABrain-One.NN-Dataset,
  author       = {Goodarzi, Arash and Kochnev, Roman and Bentyn, Zofia and Ignatov, Dmitry and Timofte, Radu},
  title        = {Neural Network Dataset},
  howpublished = {\url{https://github.com/ABrain-One/nn-dataset}},
  year         = {2024},
}
```

## Licenses

This project is distributed under the following licensing terms:
<ul><li>for neural network models adopted from other projects
  <ul>
    <li> Python code under the legacy <a href="Doc/Licenses/LICENSE-MIT-NNs.md">MIT</a> or <a href="Doc/Licenses/LICENSE-BSD-NNs.md">BSD 3-Clause</a> license</li>
    <li> models with pretrained weights under the legacy <a href="Doc/Licenses/LICENSE-DEEPSEEK-LLM-V2.md">DeepSeek LLM V2</a> license</li>
  </ul></li>
<li> all neural network models and their weights not covered by the above licenses, as well as all other files and assets in this project, are subject to the <a href="LICENSE.md">MIT license</a></li> 
</ul>

#### The idea of Dr. Dmitry Ignatov
