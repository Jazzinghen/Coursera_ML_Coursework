function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
  %LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
  %regression with multiple variables
  %   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
  %   cost of using theta as the parameter for linear regression to fit the
  %   data points in X and y. Returns the cost in J and the gradient in grad

  % Initialize some useful values
  m = length(y); % number of training examples

  % You need to return the following variables correctly
  J = 0;
  grad = zeros(size(theta));

  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the cost and gradient of regularized linear
  %               regression for a particular choice of theta.
  %
  %               You should set J to the cost and grad to the gradient.
  %

  % Compute predictions and their deltas with respect to the ground truth
  predictions = X * theta;
  deltas = predictions - y;

  % Compute theta^2 and remove the first
  regularization = theta .^ 2;
  regularization(1) = 0;

  % Mix all together
  J = (1 / (2*m)) * sum(deltas .^ 2) + (lambda / (2*m)) * sum(regularization);

  % Compute gradients

  % Create gradient regularization factor vector
  gradient_reg = (lambda / m) * theta;
  gradient_reg(1) = 0;

  % Actually compute gradient
  % Compute gradient using matrix multiplication.
  %
  % X contains all the inputs (e.g. 10 x 3, 10 examples of 3 inputs), while
  % deltas contain all the... Well, the deltas (e.g. 10 x 1, 10 deltas). To
  % compute the 1/m sum(delta * x_j) we use the transpose of inputs, allowing
  % me to do the sum of the deltas multiplied by the inputs of a specific
  % counter.
  grad = (1 / m) * X' * deltas + gradient_reg;

  % =========================================================================

  grad = grad(:);

end
