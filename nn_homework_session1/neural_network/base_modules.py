class Module(object):
    def __init__(self):
        self.output = None
        self.gradInput = None
        self.training = True

    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output
         of the module.
        """
        return self.updateOutput(input)

    def backward(self, input, gradOutput):
        """
        Performs a backpropagation step through the module,
         with respect to the given input.

        This includes
         - computing a gradient w.r.t. `input` (is needed
          for further backprop),
         - computing a gradient w.r.t. parameters (to update
          parameters while optimizing).
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput

    def updateOutput(self, inpt):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the `output` field.

        """

        pass

    def updateGradInput(self, inpt, gradOutput):
        """
        Computing the gradient of the module with respect to its own input.
        This is returned in `gradInput`. Also, the `gradInput` state variable is
        updated accordingly.

        The shape of `gradInput` is always the same as the shape of `input`.

        """

        pass

    def accGradParameters(self, inpt, gradOutput):
        """
        Computing the gradient of the module with respect to its own parameters.
        """
        pass

    def zeroGradParameters(self):
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass

    def getParameters(self):
        """
        Returns a list with its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def getGradParameters(self):
        """
        Returns a list with gradients with respect to its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def training_mode(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True

    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False

    def __repr__(self):
        """
        Pretty printing.
        """
        return "Module"


class Sequential(Module):
    """
     `input` is processed by each module (layer) in self.modules consecutively
    The resulting array is called `output`
    """

    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = []
        self.y = []

    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def updateOutput(self, inpt):
        out = inpt
        for module in self.modules:
            out = module.forward(out)
        self.output = out
        return self.output

    def backward(self, inpt, gradOutput):

        for i, module in reversed(list(enumerate(self.modules))):
            if i == 0:
                gradOutput = module.backward(inpt, gradOutput)
            else:
                gradOutput = module.backward(self.modules[i-1].output,
                                             gradOutput)
        self.gradInput = gradOutput
        return self.gradInput

    def zeroGradParameters(self):
        for module in self.modules:
            module.zeroGradParameters()

    def getParameters(self):
        """
        Should gather all parameters in a list.
        """
        return [x.getParameters() for x in self.modules]

    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [x.getGradParameters() for x in self.modules]

    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string

    def __getitem__(self, x):
        return self.modules.__getitem__(x)


class Criterion(object):

    def __init__(self):
        self.output = None
        self.gradInput = None

    def forward(self, inpt, target):
        """
        Given an input and a target, compute the loss function
        associated to the criterion and return the result.
        """
        return self.updateOutput(inpt, target)

    def backward(self, input, target):
        """
        Given an input and a target, compute the gradients of the loss function
        associated to the criterion and return the result.
        """
        return self.updateGradInput(input, target)

    def updateOutput(self, input, target):
        return self.output

    def updateGradInput(self, input, target):
        return self.gradInput

    def __repr__(self):
        """
        Pretty printing.
        """
        return "Criterion"
