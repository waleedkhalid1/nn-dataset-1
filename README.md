# Neural Network Dataset
The original version of this dataset was created by <strong>Arash Torabi Goodarzi, Roman Kochnev</strong> and <strong>Zofia Antonina Bentyn</strong> at the Computer Vision Laboratory, University of Würzburg, Germany.

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

## Environment
### Pip package manager
Create a virtual environment, activate it, and run the following command to install all the project dependencies: <br/> 
<strong>pip install -r requirements.txt</strong>

### Docker
All versions of this project are compatible with <a href='https://hub.docker.com/r/abrainone/ai-linux' target='_blank'>AI Linux</a> and can be run inside a Docker image: <br/> 
<strong> docker run -v /a/mm:&#x003C;nn-dataset path&#x003E;/ab/nn abrainone/ai-linux bash -c "PYTHONPATH=/a/mm python train.py" </strong>

## Installation of the Latest Version of NN-Dataset
pip install git+https://github.com/ABrain-One/nn-dataset

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
