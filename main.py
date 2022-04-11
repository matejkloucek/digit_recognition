import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import cv2 as cv
from tensorflow.keras.optimizers import SGD


threshold_val = 100


def on_trackbar(val):
    global threshold_val
    threshold_val = val


if __name__ == '__main__':

    # loading the data
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # normalizing the data
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)
    X_train = X_train.reshape(( X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
    print(np.shape(X_train))
    print(np.shape(y_test))

    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
        layers.Dense(10, activation='softmax')
    ])

    # compile model
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=0.001,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=3,
        callbacks=[early_stopping],
    )

    # loss, accuracy = model.evaluate(X_test, y_test)
    # print('accuracy: ', accuracy)
    # print('loss: ', loss)

    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss']].plot()
    history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
    plt.show()
    model.save('digits.model')

    loaded_model = tf.keras.models.load_model('digits.model')

    """
    for x in range(1, 7):
        img = cv.imread(f'{x}.png')[:, :, 0]
        # np.invert makes it that the numbers are black and the backround white
        img = np.invert(np.array([img]))
        print(img[:28])
        prediction = loaded_model.predict(img)
        # argmax is giving the index of the highest value thus the number with the highest probability assigned
        print('the result is probably: ', np.argmax(prediction))
        # cmap=plt.cm.binary makes it black and white
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    """
    """
    capture = cv.VideoCapture(0)
    window = cv2.namedWindow('digit_recognition')
    cv.createTrackbar('threshold', 'digit_recognition', 100, 255, on_trackbar)
    while True:
        ret, frame = capture.read()
        window = frame
        # changing the image to grayscale
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # applying threshold
        _, threshold_frame = cv.threshold(gray_frame, threshold_val, 255, cv.THRESH_BINARY)
        # cutting a frame in the center
        cv.rectangle(threshold_frame, (180, 100), (460, 380), (0, 0, 0), thickness=2)
        cut_frame = threshold_frame[100:380, 180:460]
        # resizing for 28x28
        resized_frame = cv.resize(cut_frame, (28, 28), interpolation=cv.INTER_AREA)
        # test_frame = cv.resize(resized_frame, (480, 480))
        # making it into an array and iverting
        final_frame = np.invert(np.array([resized_frame]))
        
        prediction = loaded_model.predict(final_frame)
        cv.rectangle(threshold_frame, (10, 70), (50, 10), (0, 0, 0), thickness=-1)
        cv.putText(threshold_frame, str(np.argmax(prediction)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                   (255, 255, 255), thickness=3)
        cv.imshow('digit_recognition', threshold_frame)
        if cv.waitKey(1) == ord('q'):
            break
    capture.release()
    cv.destroyAllWindows()
    """

