classdef MLP < Module

    properties
        layers (:, 1) Layer
    end

    methods
        function self = MLP(nin, nouts)
            
            arguments (Input)
                nin (1, 1) double
                nouts (:, 1) double
            end

            arguments (Output)
                self (1, 1) MLP
            end

            sz = [nin; nouts];

            depth = numel(nouts);
            self.layers = arrayfun(@(idx) Layer(sz(idx), sz(idx + 1), "nonlin", idx ~= depth), 1:depth);

        end

        function act = fire(self, x)

            arguments (Input)
                self (1, 1) MLP
                x (:, 1) Value
            end

            arguments (Output)
                act (:, 1) Value
            end

            for idx = 1:numel(self.layers)
                layer = self.layers(idx);
                x = layer.fire(x);
            end
            
            act = x;

        end

        function params = parameters(self)

            arguments (Input)
                self (1, 1) MLP
            end

            arguments (Output)
                params (:, 1) Value
            end

            params = arrayfun(@(layer) layer.parameters(), self.layers, "UniformOutput", false);
            params = cat(1, params{:});

        end
    end
end