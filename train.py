import tensorflow as tf
import time
import numpy as np

from dataloader import MicrowaveDataloader
from model import MicrowaveLSTM


def main():
    window_size = 1000
    num_classes = 2
    training_percentage = 0.7
    dataset_source = './data/microwave_dataset'
    weights_save_location = './weights/MicrowaveLSTM' + str(window_size)

    dataloader = MicrowaveDataloader(dataset_source, window_size, training_percentage)

    lstm = MicrowaveLSTM(window_size, num_classes, weights_save_location)
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    max_epochs = 200
    val_break_lim = 5

    total_start = time.time()
    val_break_count = 0
    val_sum = 0
    step_count = 0
    print('Establishing base validation loss...')
    init_iou_list = []
    for step, (x_batch_val, y_batch_val) in enumerate(dataloader.val_set):
        # visualize_view(x_batch_val)
        # visualize_grid_mlab(y_batch_val)
        pred = auto_encoder.tf_model(x_batch_val, training=False)
        loss_value = loss_fn(y_batch_val, pred)
        val_sum += loss_value

        init_iou_list.append(find_iou(pred, y_batch_val))

        step_count += 1
    val_loss = val_sum / step_count
    print('Base validation loss:', float(val_loss))

    iou_array = np.array(init_iou_list)
    ave_iou = np.sum(iou_array) / len(init_iou_list)
    print('Base Validation IoU:', ave_iou)

    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("Starting epoch,", epoch)

        train_iou_list = []
        for step, (x_batch_train, y_batch_train) in enumerate(dataloader.train_set):
            with tf.GradientTape() as tape:
                pred = auto_encoder.tf_model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, pred)

                grads = tape.gradient(loss_value, auto_encoder.tf_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, auto_encoder.tf_model.trainable_weights))

                train_iou_list.append(find_iou(pred, y_batch_train))

            if step % 100 == 0:
                print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))

        iou_array = np.array(train_iou_list)
        ave_iou = np.sum(iou_array) / len(train_iou_list)
        print('Average IoU for training set:', ave_iou)
        val_sum = 0
        step_count = 0
        print('Validating epoch...')
        val_iou_list = []
        for step, (x_batch_val, y_batch_val) in enumerate(dataloader.val_set):
            pred = auto_encoder.tf_model(x_batch_val, training=False)
            loss_value = loss_fn(y_batch_val, pred)
            val_sum += loss_value

            val_iou_list.append(find_iou(pred, y_batch_val))

            step_count += 1
        val_ave = val_sum / step_count
        print('Validation loss for epoch:', float(val_ave))
        iou_array = np.array(val_iou_list)
        ave_iou = np.sum(iou_array) / len(val_iou_list)
        print('Average IoU for validation set:', ave_iou)
        if val_ave >= val_loss:
            val_break_count += 1
            print('Validation error did not decrease, this has occurred for', val_break_count, 'epoch(s)...')
            if val_break_count >= val_break_lim:
                print('Ending Training early, reached minimum validation error...')
                break
        else:
            val_break_count = 0
            val_loss = val_ave
            auto_encoder.tf_model.save_weights(weights_save_location)
        print('Epoch time:', time.time() - epoch_start)
        print('Cumulative time:', time.time() - total_start)
    print('Training complete!')


if __name__ == '__main__':
    main()
