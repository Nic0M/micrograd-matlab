classdef Neuron < Module

    properties
        w (:, 1) Value % weights
        b Value {mustBeScalarOrEmpty} = Value.empty % bias
        nonlin (1, 1) logical
    end

    methods

        function self = Neuron(nin, kwargs)
            % Neuron Creates a Neuron with nin inputs.

            arguments (Input)
                nin (1, 1) double
                kwargs.nonlin (1, 1) logical = true
            end

            arguments (Output)
                self (1, 1) Neuron
            end

            if nin == 0
                self.w = Value.empty;
            else
                self.w = arrayfun(@(x) Value(x), 2 * rand(nin, 1) - 1);
            end
            self.b = Value(0);
            self.nonlin = kwargs.nonlin;

        end

        function act = fire(self, x)
            % fire Determines whether the Neuron should fire or not.
            %
            % If non-linear activation, uses ReLU function.

            arguments (Input)
                self (1, 1) Neuron
                x (:, 1) Value
            end

            arguments (Output)
                act (1, 1) Value
            end

            synapses = arrayfun(@(wi, xi) wi * xi, self.w, x);
            act = Value.sum(synapses) + self.b;

            if self.nonlin
                act = act.relu();
            end

        end

        function params = parameters(self)
            % parameters Returns an array of all of the parameters in the
            % current Neuron

            arguments (Input)
                self (1, 1) Neuron
            end

            arguments (Output)
                params (:, 1) Value
            end

            params = [self.w; self.b];
        end
    end
end