from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import models,layers
import time



def main():
    (train_X,train_Y),(test_X,test_Y) = mnist.load_data()
    print("trainX : "+str(train_X.shape) )
    print("trainY : "+str(train_Y.shape) )
    print("testX : "+str(test_X.shape) )
    print("testX : "+str(test_Y.shape) )

    start = 0 # index of first image to be plotted
    for i in range(start,start+9):
        plt.subplot(330+1+(i%9))
        plt.imshow(train_X[i], cmap = plt.get_cmap("gray"))
    #plt.show()

    start_time = time.time()

    model = models.Sequential()
    model.add(layers.Flatten(input_shape = (28,28)))
    model.add(layers.Dense(16,activation = "relu"))
    model.add(layers.Dense(16,activation = "relu"))
    model.add(layers.Dense(10,activation = "softmax"))

    print(model.summary())

    model.compile(optimizer = "adam",
                    loss = "sparse_categorical_crossentropy",metrics = ["accuracy"])
    model.fit(train_X,train_Y,epochs=25,batch_size=128)

    print("--- %s seconds ---" % (time.time() - start_time))

    # print("Evaluate on test data")
    # results = model.evaluate(X_test, Y_test, batch_size=128)
    # print("test loss, test acc:", results)

    # # Generate predictions (probabilities -- the output of the last layer)
    # # on new data using `predict`
    # print("Generate predictions for 3 samples")
    # predictions = model.predict(x_test[:3])
    # print("predictions shape:", predictions.shape)



if __name__ ==  "__main__" :
    main()
else :
    print("NN_TF can't be accessed")

