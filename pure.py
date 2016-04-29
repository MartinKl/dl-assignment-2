import nltk
from nltk.corpus import brown

import blocks
from blocks.bricks import MLP, Softmax, Rectifier, Linear
from blocks.initialization import Constant, Uniform, IsotropicGaussian
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.bricks.lookup import LookupTable
from blocks.roles import WEIGHT

import theano
from theano import tensor

# ------------------------------------------------------------- #

words = brown.words()
V = list(set(words))
v = len(V)

table = LookupTable(length=v, dim=1, weights_init=IsotropicGaussian(0.01), biases_init=Constant(0))

PARAM_H_SIZE = 100

x = tensor.matrix('x',dtype="int64")
y = tensor.lvector('y')

in_to_h = table.apply(x)
# now average!
x.mean(axis=1) # figure out what that really does, why axis 1?
#
h_to_out = Linear(name='h_to_out', input_dim=PARAM_H_SIZE, output_dim=v,
                  weights_init=IsotropicGaussian(0.01), biases_init=Constant(0))
y_hat = Softmax().apply(h_to_out.apply(x))

cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
# is there something more to do with cost?
cg = ComputationGraph(cost)



train_set = # TODO
data_stream = Flatten(DataStream.default_stream(train_set,
                                                iteration_scheme=SequentialScheme(train_set.num_examples, batch_size=256)))

algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                     step_rule=Scale(learning_rate=0.1))

test_set = # TODO
test_stream = Flatten(DataStream.default_stream(test_set, 
                                                iteration_scheme=SequentialScheme(test_set.num_examples, batch_size=1024)))

monitor = DataStreamMonitoring(variables=[cost], data_stream=test_stream, prefix="test")
main_loop = MainLoop(data_stream=data_stream, algorithm=algorithm, extensions=[monitor, FinishAfter(after_n_epochs=1), Printing()])
main_loop.run()
