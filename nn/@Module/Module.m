classdef Module

    methods (Abstract)
        params = parameters(self)
    end

    methods
        function zero_grad(self)
            params = self.parameters();
            for idx = 1:numel(params)
                p = params(idx);
                p.grad = 0;
            end
        end
    end

end