2. How many dimensions must the inputs of an RNN layer have? What about its outputs?

Input is like: Batch_Size * Num_Of_Steps * Embedding_Dimension. Output is the same only that batch size and num of steps don't differ but embedding size may differ and is equal to the number of neurons in the last layer.

3. If you want to build a sequence to sequence RNN, which RNN layers should have return_sequence=True? all of them. What about a sequence to vector RNN? all except last one.
5. 
