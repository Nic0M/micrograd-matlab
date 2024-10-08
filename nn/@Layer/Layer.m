classdef Layer < Module

    properties
        neurons (:, 1) Neuron
    end

    methods
        function self = Layer(nin, nout, kwargs)
            % Layer Constructor
            %
            % nin - number of inputs
            % nout - number of outputs
            % nonlin - use non-linear activation function

            arguments (Input)
                nin (1, 1) double
                nout (1, 1) double
                kwargs.nonlin (1, 1) logical = true
            end

            arguments (Output)
                self (1, 1) Layer
            end

            self.neurons = arrayfun(@(foo) Neuron(nin, "nonlin", kwargs.nonlin), 1:nout);
        end

        function act = fire(self, x)
            % fire Determines whether each Neuron in the layer should fire.

            arguments (Input)
                self (1, 1) Layer
                x (:, 1) Value
            end

            arguments (Output)
                act (:, 1) Value
            end

            act = arrayfun(@(n) n.fire(x), self.neurons);
        end

        function params = parameters(self)
            % parameters Returns an array of all of the parameters in the
            % current layer

            arguments (Input)
                self (1, 1) Layer
            end

            arguments (Output)
                params (:, 1) Value
            end

            params = arrayfun(@(n) n.parameters(), self.neurons, "UniformOutput", false);
            params = cat(1, params{:});
        end
    end
end