# Flexible Neural Network

- This code is for making neural network which change the input during run time.

- By using the tf.cond function, we select which input node we will use to output.

Example in code:

There are three variables (x, y, z) and I want to sum some of this depending on the check flag.
For example, when check is [T, F, T], then I want to make output of  (x + z).
