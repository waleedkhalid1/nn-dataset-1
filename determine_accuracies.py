import json
import os

import torchvision
from torchvision.transforms import transforms

from CVModelEvaluator.CVModelEvaluator import CVModelEvaluator


number_of_epochs = 2


def main():
    transform = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True,
        download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False,
        download=True, transform=transform
    )

    for model in os.listdir("./Dataset"):
        model = str(os.fsdecode(model))
        if os.path.isdir("./Dataset/" + model):
            try:
                evaluator = CVModelEvaluator("Dataset." + model, train_set, test_set)
                accuracy = evaluator.evaluate(number_of_epochs)
                accuracies = {
                    str(evaluator.get_args()): (accuracy, number_of_epochs)
                }
                with open("./Dataset/" + model + "/accuracies.json", "w+") as acc_file:
                    json.dump(accuracies, acc_file)
            except Exception as error:
                print("failed to determine accuracy for", model)
                with open("./Dataset/" + model + "/error.txt", "w+") as error_file:
                    error_file.write(str(error))


if __name__ == "__main__":
    main()
