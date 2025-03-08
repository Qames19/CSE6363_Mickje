########################################################################
# Student:      James Mick                                             #
# ID:           2234431                                                #
# Course:       CSE-6363                                               #
# Assignment:   2                                                      #
# Professor:    Dr. Alex Dilhoff                                       #
# Assisted by:  ChatGPT o3-mini-high                                   #
########################################################################

class Layer:
    def forward(self, inputs):
        """
        Performs a forward pass and returns the output

        Parameters:
        inputs (numpy.ndarray): The input data.

        Returns:
        numpy.ndarray:  The output of the layer.
        """
        
        # Return an error if the subclass does not define the implementation of forward
        raise NotImplementedError("Forward method not implemented in {self.__class__.__name__}.")
    
    def backward(self, grad_output):
        """
        Performs a backward pass and returns the output

        Parameters:
        grad_output (numpy.ndarray): The gradient of the loss with respect to the layer's output.

        Returns:
        numpy.ndarray:  The gradient of the loss with respect to the layer's input.
        """

        # Return an error if the subclass does not define the implementation of backward
        raise NotImplementedError("Backward method not implemented in {self.__class__.__name__}.")
