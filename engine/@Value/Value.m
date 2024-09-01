classdef Value < handle
    % Value Stores a single scalar value and its gradient
    
    properties
        data (1, 1) double
        grad (1, 1) double = 0
        backward_closure (1, 1) function_handle = @nop
        prev (1, 1) dictionary = dictionary(Value.empty, false) % this is a little bit of a hack to get around the error message: "Unable to perform a lookup in a dictionary with unset key and value types."
        op (1, 1) string = ""
    end

    methods
        function self = Value(data, children, op) %#codegen

            arguments (Input)
                data (1, 1) double
                children (:, 1) Value = Value.empty
                op (1, 1) string = ""
            end

            arguments (Output)
                self (1, 1) Value
            end

            self.data = data;
            
            for idx = 1:numel(children)
                child = children(idx);
                self.prev(child) = true;
            end

            self.op = op;
        end

        function out = plus(self, other) %#codegen

            arguments (Input)
                self (1, 1) Value
                other (1, 1) Value
            end

            arguments (Output)
                out (1, 1) Value
            end

            out = Value(self.data + other.data, [self, other], "+");
          
            function backward_closure()
                self.grad = self.grad + out.grad;
                other.grad = other.grad + out.grad;
            end
            out.backward_closure = @backward_closure;
        end

        function out = mtimes(self, other) %#codegen

            arguments (Input)
                self (1, 1) Value
                other (1, 1) Value
            end

            arguments (Output)
                out (1, 1) Value
            end

            out = Value(self.data * other.data, [self, other], "*");

            function backward_closure()
                self.grad = self.grad + other.data * out.grad;
                other.grad = other.grad + self.data * out.grad;
            end
            out.backward_closure = @backward_closure;
        end

        function out = mpower(self, other) %#codegen

            arguments (Input)
                self (1, 1) Value
                other (1, 1) double
            end
            
            arguments (Output)
                out (1, 1) Value
            end

            out = Value(self.data ^ other, self, "^" + num2str(other));

            function backward_closure()
                self.grad = self.grad + (other * self.data ^ (other - 1)) * out.grad;
            end
            out.backward_closure = @backward_closure;
        end

        function out = relu(self) %#codegen

            arguments (Input)
                self (1, 1) Value
            end

            arguments (Output)
                out (1, 1) Value
            end
            
            out = Value(max(0, self.data), self, "ReLU");

            function backward_closure()
                self.grad = self.grad + (out.data > 0) * out.grad; % evil: using math with booleans
            end
            out.backward_closure = @backward_closure;
        end

        function backward(self) %#codegen

            % Implement simple amortized vector for speed
            capacity = 16;
            topo = repmat(Value(0), capacity, 1);
            tail = 0;

            % Topological order of all of the children in the graph
            visited = dictionary(Value.empty, false);
            function build_topo(v)

                if ~visited.lookup(v, "FallbackValue", false)
                    visited(v) = true;
                    children = v.prev.keys();
                    for idx_child = 1:numel(children)
                        child = children(idx_child);
                        build_topo(child)
                    end
                    tail = tail + 1;
                    if capacity < tail
                        % Allocate new array with twice the size and copy old elemnts
                        tmp = topo;
                        capacity = capacity * 2;
                        topo = repmat(Value.empty, capacity, 1);
                        topo(1:tail - 1) = tmp;
                    end
                    topo(tail) = v;
                end
            end
            build_topo(self);
            % Trim any extra allocated memory
            topo = topo(1:tail);

            self.grad = 1;
            for idx_topo = numel(topo):-1:1
                v = topo(idx_topo);
                v.backward_closure();
            end
        end

        function out = uminus(self) %#codegen
            out = self * -1;
        end

        function out = minus(self, other) %#codegen
            out = self + (-other);
        end

        function out = mrdivide(self, other) %#codegen
            out = self * other^(-1);
        end
    end

    methods (Static)
        function out = sum(values)
            % sum Convenience function to sum an array of Value objects
            
            arguments (Input)
                values Value
            end

            arguments (Output)
                out (1, 1) Value
            end

            values = values(:);

            while numel(values) > 1
                % If odd number of elements, add a zero-value element
                if mod(numel(values), 2) == 1
                    values = [values; Value(0)];
                end

                newvalues = repmat(Value(0), numel(values) / 2, 1);

                for idx = 1:2:numel(values)
                    newvalues((idx + 1) / 2) = values(idx) + values(idx + 1);
                end
                values = newvalues;
            end

            out = values;
        end
    end
end

function nop()
end