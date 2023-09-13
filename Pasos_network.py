'''
Se importa el dataset de MNIST
'''
import mnist_loader # llama al módulo mnist_loader que desempaqueta el dataset y transforma las imágenes de 28*28 en matrices de 784*1
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

'''
Se importa la red:
    784 neuronas de entrada
    30 neuronas capa de en medio
    10 neuronas de salida
'''
import network
net = network.Network([784, 30, 10])

'''
SGD:
    SGD(self, training_data, epochs, mini_batch_size, eta)
    30 épocas
    10 mini-batch size
    3.0 tasa de aprendizaje
'''
net.SGD(training_data, 30, 10, 0.0010, test_data=test_data)