function [X, y] = make_moons(kwargs)
% make_moons Make two interleaving half circles.
%
% A simple toy dataset to visualize clustering and classification
% algorithms.
%
% https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html#sklearn.datasets.make_moons
%
% BSD 3-Clause License
% 
% Copyright (c) 2007-2024 The scikit-learn developers.
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 
% * Redistributions of source code must retain the above copyright notice, this
%   list of conditions and the following disclaimer.
% 
% * Redistributions in binary form must reproduce the above copyright notice,
%   this list of conditions and the following disclaimer in the documentation
%   and/or other materials provided with the distribution.
% 
% * Neither the name of the copyright holder nor the names of its
%   contributors may be used to endorse or promote products derived from
%   this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

arguments (Input)
    kwargs.n_samples (:, 1) int32 = 100
    kwargs.shuffle (1, 1) logical = true
    kwargs.noise double {mustBeScalarOrEmpty} = []
    kwargs.random_state double {mustBeScalarOrEmpty} = []
end

arguments (Output)
    X (:, 2) % the generated samples
    y (:, 1) % the integer labels (0 ori 1) for class membership of each sample
end

n_samples = kwargs.n_samples;

if isscalar(n_samples)
    n_samples_out = idivide(n_samples, 2);
    n_samples_in = n_samples - n_samples_out;
else
    if numel(n_samples) == 2
        n_samples_out = n_samples(1);
        n_samples_in = n_samples(2);
    else
        error("`n_samples` can either be an int or a two-element array.")
    end
end

if ~isempty(kwargs.random_state)
    rng(kwargs.random_state)
end

outer_circ_x = cos(linspace(0, pi, n_samples_out));
outer_circ_y = sin(linspace(0, pi, n_samples_out));
inner_circ_x = 1 - cos(linspace(0, pi, n_samples_in));
inner_circ_y = 1 - sin(linspace(0, pi, n_samples_in)) - 0.5;

X = horzcat([outer_circ_x'; inner_circ_x'], [outer_circ_y'; inner_circ_y']);
y = [zeros(n_samples_out, 1); ones(n_samples_in, 1)];

if kwargs.shuffle
    indices = randperm(length(X));
    X = X(indices, :);
    y = y(indices);
end

if ~isempty(kwargs.noise)
    X = X + random("Normal", 0, kwargs.noise, size(X));
end

end