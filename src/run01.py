import mnist_loader
import network_matrix_based

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# print "Non-matrix-based approach:"
# net = network.Network([784, 10, 10])
# #net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
# net.SGD(training_data, 10, 10, 3.0, test_data=test_data)

print

print "Matrix-based network:"
net_matrix_based = network_matrix_based.Network([784, 10, 10])
net_matrix_based.SGD(training_data, 30, 10, 3.0, test_data=test_data)
