# Flexible Neural Network

#### flexible_nn.py

- This code is for making neural network which change the input during run time.

- By using the tf.cond function, we select which input node we will use to output.

Example in code:

There are three variables (x, y, z) and I want to sum some of this depending on the check flag.
For example, when check is [T, F, T], then I want to make output of  (x + z).


#### multi_input.py

- This code is for making neural network whose input comes from different output.
- By using tf.concat() function, multiple placeholder can be merged and be input for the neural network