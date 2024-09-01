# micrograd-matlab

Implementation of [karpathy/micrograd](https://github.com/karpathy/micrograd) in MATLAB.

## Usage
Mimics the usage of the original Python micrograd.
The notable exceptions are the lack of the += operator in MATLAB, and the lack of support for method chaining on arithmetic operations.
Additionally, MATLAB uses the ^ operator instead of ** for exponentiation.
Finally, only the matrix operators have been overloaded, and the elementwise dot operators are not supported.

```matlab
a = Value(-4);
b = Value(2);
c = a + b;
d = a * b + b^3;
c = c + c + 1;
c = c + 1 + c + (-a);
tmp1 = b + a;
d = d + d * 2 + tmp1.relu();
tmp2 = b - a;
d = d + 3 * d + tmp2.relu();
e = c - d;
f = e ^ 2;
g = f / 2;
g = g + 10 / f;
fprintf("g.data = %.4f\n", g.data); % prints 24.7041, the outcome of this forward pass
g.backward();
fprintf("a.grad = %.4f\n", a.grad); % prints 138.8338, i.e. the numerical value of dg/da
fprintf("b.grad = %.4f\n", b.grad); % prints 645.5773, i.e. the numerical value of dg/db
```

## Training a neural net

The script `demo.mlx` provides an equivalent demo of the original micrograd of training a neural network (MLP) binary classifier with 2 16-node hidden layers.
The classification after 100 iterations gives the following decision boundary:
![Decision boundary after training on moon dataset](/images/trained.png)