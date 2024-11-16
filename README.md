# Neural Network Dataset
The original version of this dataset was created by the team of <strong>Arash Torabi Goodarzi, Roman Kochnev</strong> and <strong>Zofia Antonina Bentyn</strong>.

## Contribution

To Add more models to the dataset, the following criteria must be met.
1. Each Model must be saved in a separated directory inside the initially provided directory named "Dataset", next to other already provided models.
2. The code for each model must be provided in the respective directory in a python file named "code.py"
3. The main class for each model must be named "Net".
4. The required arguments to initialize the "Net" class must be stored in a list inside a separate python file named "args.py" next to "code.py". If no arguments are needed, provide an empty python list in the "args.py" file.

For examples, see the models in the Dataset directory.

## Environment

In addition to pip/conda package managers, all versions of this project are compatible with <a href='https://hub.docker.com/r/abrainone/ai-linux' target='_blank'>AI Linux</a> and can be run inside a Docker image <strong>abrainone/ai-linux</strong> with command: <br/> 
<strong> docker run -v /a/mm:&#x003C;nn-dataset path&#x003E; abrainone/ai-linux bash -c "PYTHONPATH=/a/mm python determine_accuracies.py" </strong>

## Citation

If you find Neural Network Dataset to be useful for your research, please consider citing:
```bibtex
@misc{nn-dataset,
  author       = {Goodarzi, Arash and Kochnev, Roman and Bentyn, Zofia and Ignatov, Dmitry},
  title        = {Neural Network Dataset},
  howpublished = {\url{https://github.com/ABrain-One/nn-dataset}},
  year         = {2024},
}
```
