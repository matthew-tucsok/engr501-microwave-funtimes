"""
ENGR 501 Course Project
Matthew Tucsok 37924164
Jack McClelland 31046162

Training script for our various models. To change models, modify which class is used when creating the instance of
dl_model.
"""

import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt

from dataloader import MicrowaveDataloader
from model import ConvNet, FCNet, TinyFCNet, TinyConvNet  # All models imported for convenience


def main():
    dataset_source = './data/microwave_dataset'  # Path to dataset
    weights_name = 'TinyConvNet1'  # This should be changed for each network
    weights_save_location = './weights/' + weights_name

    dataloader = MicrowaveDataloader(source=dataset_source)  # Creation of normalized train/val/test sets for TF loop

    dl_model = TinyConvNet(weights_save_location)  # Creating the DL model. Change the class to change models.
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    max_epochs = 100000  # Arbitrarily large to ensure training to completion
    total_epoch_count = 0
    val_break_lim = 10  # Stopping criterion 1. If validation error does not decrease for 10 epochs in a row, end training
    break_epsilon = 0.0001  # If validation error does not decrease fast enough 10 epochs in a row, end training

    total_start = time.time()
    cumulative_time = 0
    val_break_count = 0
    val_sum = 0
    step_count = 0
    mistakes_total = 0

    ave_val_losses = []
    ave_val_accuracies = []

    """
    This loop is to determine what the starting accuracy and loss is. This architecture is repeated multiple times
    """
    print('Establishing base validation statistics before training...')
    for step, (x_batch_val, y_batch_val) in enumerate(dataloader.val_set):  # This format is to allow mini-batches
        pred = dl_model.tf_model(x_batch_val, training=False)  # Predicting label
        label_pred = np.round(np.array(pred))  # Converting to numpy-friendly datatype
        mistake = np.abs(label_pred - y_batch_val)  # If 1, then the pred and actual label do not match
        mistakes_total += int(mistake)
        loss_value = loss_fn(y_batch_val, pred)  # Using binary-cross-entropy as loss_fn
        val_sum += loss_value
        step_count += 1
    ave_val_loss = val_sum / step_count
    ave_accuracy = (step_count - mistakes_total) / step_count
    ave_val_losses.append(ave_val_loss)
    ave_val_accuracies.append(ave_accuracy)
    print('Base average validation loss:', float(ave_val_loss))
    print('Base average validation accuracy:', float(ave_accuracy))
    total_epoch_count += 1

    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("Starting epoch,", epoch)

        step_count = 0
        for step, (x_batch_train, y_batch_train) in enumerate(dataloader.train_set):
            with tf.GradientTape() as tape:  # This is required when creating a custom training loop
                pred = dl_model.tf_model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, pred)
                grads = tape.gradient(loss_value, dl_model.tf_model.trainable_weights)  # Calculating gradients
                optimizer.apply_gradients(zip(grads, dl_model.tf_model.trainable_weights))  # Using SGD to update weights
                step_count += 1
            if step % 1 == 1:
                print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
        val_sum = 0
        mistakes_total = 0
        step_count = 0
        print('Validating epoch...')
        for step, (x_batch_val, y_batch_val) in enumerate(dataloader.val_set):
            pred = dl_model.tf_model(x_batch_val, training=False)
            label_pred = np.round(np.array(pred))
            mistake = np.abs(label_pred - y_batch_val)
            mistakes_total += int(mistake)
            loss_value = loss_fn(y_batch_val, pred)
            val_sum += loss_value
            step_count += 1
        val_ave = val_sum / step_count
        ave_accuracy = (step_count - mistakes_total) / step_count
        ave_val_losses.append(val_ave)
        ave_val_accuracies.append(ave_accuracy)
        total_epoch_count += 1
        print('Average validation loss for epoch', epoch, ':', float(val_ave))
        print('Average accuracy for epoch', epoch, ':', ave_accuracy)
        if val_ave >= ave_val_loss or abs(val_ave - ave_val_loss) < break_epsilon:  # If stop criterion met, count it
            val_break_count += 1
            print('Average validation error did not decrease, this has occurred for', val_break_count, 'epoch(s)...')
            if val_break_count >= val_break_lim:  # If stop criterion reached 10 times, stop training
                print('Ending Training early, reached minimum average validation error...')
                break
        else:
            val_break_count = 0
            ave_val_loss = val_ave
            dl_model.tf_model.save_weights(weights_save_location)  # Notice weights are only saved if stop criterion not met
        print('Epoch time:', time.time() - epoch_start)
        cumulative_time = time.time() - total_start
        print('Cumulative time:', cumulative_time)

    print('Training complete!')

    with open('./training_results/' + weights_name + '_losses.txt', 'w') as f:
        for loss in ave_val_losses:
            f.write(str(float(loss)))
            f.write('\n')

    with open('./training_results/' + weights_name + '_accuracies.txt', 'w') as f:
        for accuracy in ave_val_accuracies:
            f.write(str(accuracy))
            f.write('\n')

    with open('./training_results/' + weights_name + 'total_time.txt', 'w') as f:
        f.write(str(cumulative_time))
        f.write('\n')

    plt.plot(range(total_epoch_count), ave_val_losses, label='Ave training loss')
    plt.title('Average Training Loss on Validation Set vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.show()
    plt.savefig('Average_training_loss')

    plt.plot(range(total_epoch_count), ave_val_accuracies, label='Ave training Accuracy')
    plt.title('Average Training Accuracy on Validation Set vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Average Accuracy')
    plt.show()
    plt.savefig('Average_training_accuracy')

    # Running final accuracy test on test set, which is not involved whatsoever during training
    mistakes_total = 0
    step_count = 0
    for step, (x_batch_test, y_batch_test) in enumerate(dataloader.test_set):
        pred = dl_model.tf_model(x_batch_test, training=False)
        label_pred = np.round(np.array(pred))
        mistake = np.abs(label_pred - y_batch_test)
        mistakes_total += int(mistake)
        step_count += 1

    ave_accuracy = (step_count - mistakes_total) / step_count
    print('Final accuracy on test set:', ave_accuracy)


if __name__ == '__main__':
    main()
