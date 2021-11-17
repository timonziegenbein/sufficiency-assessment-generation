from keras.callbacks import Callback


class CaptureBestEpoch(Callback):

    test_x = None
    test_y = None

    best_val_accuracy = 0.0

    best_predictions = None

    def __init__(self, test_x, test_y):
        super(Callback, self).__init__()

        self.test_x = test_x
        self.test_y = test_y

    def on_epoch_end(self, epoch, logs={}):
        if logs['val_acc'] >= self.best_val_accuracy:
            self.best_val_accuracy = logs['val_acc']
            self.best_epoch = epoch
            self.best_predictions = self.model.predict_classes(self.test_x, verbose=False)
