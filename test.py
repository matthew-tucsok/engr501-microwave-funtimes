"""
ENGR 501 Course Project
Matthew Tucsok 37924164
Jack McClelland 31046162

Test script for our various models. To change models, modify which class is used when creating the instance of
dl_model. We used this script to see what the accuracy on the test set was at for any model asynchronous of training.
"""
import numpy as np

from dataloader import MicrowaveDataloader
from model import ConvNet, FCNet, TinyFCNet, TinyConvNet


def main():
    dataset_source = './data/microwave_dataset'
    weights_save_location = './weights/ConvNetModified1'

    dataloader = MicrowaveDataloader(source=dataset_source)

    dl_model = ConvNet(weights_save_location)

    mistakes_total = 0
    step_count = 0
    for step, (x_batch_test, y_batch_test) in enumerate(dataloader.test_set):
        pred = dl_model.tf_model(x_batch_test, training=False)
        label_pred = np.round(np.array(pred))
        mistake = np.abs(label_pred - y_batch_test)
        mistakes_total += int(mistake)
        step_count += 1

    accuracy = (step_count - mistakes_total) / step_count
    print('Final accuracy on test set:', accuracy)


if __name__ == '__main__':
    main()
