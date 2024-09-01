clear; close all; clc;

addpath("engine");
addpath("nn");

rng(1337);

[X, y] = make_moons(n_samples=100, noise=0.1);
y = y*2 - 1; % make y be -1 or 1
% visualize in 2D
figure();
scatter(X(:, 1), X(:, 2), 20, y, "filled")
colormap("jet");
title("Sklearn Moons Dataset")

% initialize a model
model = MLP(2, [16, 16, 1]); % 2-layer neural network
display(model);
fprintf("number of parameters %d\n", length(model.parameters()));

plot_decision(model, X, y);

% loss function
function [total_loss, acc] = loss(model, X, y, batch_size)

    arguments (Input)
        model (1, 1) MLP
        X (:, 2) double
        y (:, 1) double
        batch_size double {mustBeScalarOrEmpty, mustBeInteger} = []
    end

    arguments (Output)
        total_loss (1, 1) Value
        acc (1, 1) double
    end

    if isempty(batch_size)
        Xb = X;
        yb = y;
        batch_size = length(X);
    else
        ri = randperm(size(X, 1));
        ri = ri(1:batch_size);
        Xb = X(ri);
        yb = y(ri);
    end
    
    % Convert from floats/doubles to Value objects
    inputs = arrayfun(@(data) Value(data), Xb);
    
    scores = repmat(Value(0), batch_size, 1);
    for idx_point = 1:batch_size
        point = inputs(idx_point, :);
        scores(idx_point) = model.fire(point');
    end

    % svm "max-margin" loss
    losses = arrayfun(@(yi, scorei) (1 + -yi*scorei), yb, scores);
    losses = arrayfun(@(loss) loss.relu(), losses); % need to apply ReLU separately since MATLAB doesn't support (1 + -yi*scorei).relu() syntax 
    data_loss = Value.sum(losses) * (1.0 / batch_size);
    % L2 regularization
    alpha = 1e-4;
    reg_loss = alpha * Value.sum(arrayfun(@(p) p * p, model.parameters()));
    total_loss = data_loss + reg_loss;

    % also get accuracy
    accuracy = arrayfun(@(yi, scorei) (yi > 0) == (scorei.data > 0), yb, scores);
    acc = sum(accuracy) / batch_size;
end
[total_loss, acc] = loss(model, X, y);
display(total_loss);
display(acc);

% optimization
epochs = 100;
accs = zeros(epochs, 1);
for k = 0:(epochs - 1)

    % forward
    [total_loss, acc] = loss(model, X, y);

    % backward
    model.zero_grad();
    total_loss.backward();

    % update (sgd)
    learning_rate = 1.0 - 0.9*k/100;
    params = model.parameters();
    for idx_p = 1:numel(params)
        p = params(idx_p);
        p.data = p.data - learning_rate * p.grad;
    end

    if mod(k, 1) == 0
        fprintf("step %d loss %.4f, accuracy %.1f%%\n", k, total_loss.data, acc * 100);
    end

    accs(k + 1) = acc;

end

figure()
plot(1:epoch, accs);
xlabel("Epoch");
ylabel("Accuracy (%)");
title("Training Accuracy Over Time");

plot_decision(model, X, y);

function plot_decision(model, X, y)
h = 0.25;
x_min = min(X(:, 1)) - 1;
x_max = max(X(:, 1)) + 1;
y_min = min(X(:, 2)) - 1;
y_max = max(X(:, 2)) + 1;
[xx, yy] = meshgrid(x_min:h:x_max, y_min:h:y_max);
Xmesh = [xx(:), yy(:)];
inputs = arrayfun(@(data) Value(data), Xmesh);
scores = repmat(Value.empty, length(Xmesh), 0);
for idx_point = 1:length(Xmesh)
    point = inputs(idx_point, :);
    scores(idx_point) = model.fire(point');
end
Z = arrayfun(@(s) s.data > 0, scores);
Z = reshape(Z, size(xx));
Z = Z*2 - 1;

figure(); hold on;
contourf(xx, yy, Z, "FaceAlpha", 0.8);
scatter(X(:, 1), X(:, 2), 40, y, "filled");
xlim([min(xx, [], "all"), max(xx, [], "all")]);
ylim([min(yy, [], "all"), max(yy, [], "all")]);
colormap("jet");
end

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