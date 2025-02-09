# Resources

## High Level Overview

This is to give a fundamental view of any deep learning system. All that goes on at any point in time is the process described below. Now I am including a bit of technical details, so that it sparks your curiousity, but trying to keep it simple so you don't get stuck in a rabbit hole.

Here is a humble view of the stack.

* **Data** - Now what we are doing is data-driven learning, so we need data in some machine readable form. For image it's pixels, for text it's tokens etc. Now we can do transformations to condense information here (for another day.)

* **Neural Networks (NN)** - Very simply imagine a graph of operations, which operate on the input data and produce an output. We will get to the details of operations later on. For now assume add, mult, sub, exp etc.

* **Ground Truth** - We had data. We passed it through the fancy NN. We have a output. Now this output for a image classfier will be label of the image. So there is a ground truth label and our NN output.

* **Backprop** - Nothing fancy. Our NN is not as smart as our brain. So probably it has not classified the image we wanted correctly. There is an error. We capture this error and propagate it back through the neural network and update some of its parameters.

* **Iterate** - We continue this process repeatedly till we find the optimal set of parameters.

Comparision with Human Brain - You are driving (see image -> brain encodes -> driving NN makes a decision -> Your decision kills a dog -> Error -> Trauma in the brain -> Rewiring the NN (brain) -> ... -> F1 Driver.)

Cool. Hope you get the intro. Lets get a bit into the tech.
