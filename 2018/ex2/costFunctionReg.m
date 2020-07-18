function [J, grad] = costFunctionReg(theta, X, y, lambda)
  %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
  %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
  %   theta as the parameter for regularized logistic regression and the
  %   gradient of the cost w.r.t. to the parameters.

  % Initialize some useful values
  m = length(y); % number of training examples

  % You need to return the following variables correctly
  J = 0;
  grad = zeros(size(theta));

  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the cost of a particular choice of theta.
  %               You should set J to the cost.
  %               Compute the partial derivatives and set grad to the partial
  %               derivatives of the cost w.r.t. each parameter in theta



  % Compute the the entire outups by using matrix multiplication.
  sigmoid_input = X * theta;
  sigmoid_computed = sigmoid(sigmoid_input);

  % Compute cost
  % Added the sum of all thetas but the first one
  J = -1/m * (y' * (log(sigmoid_computed)) + ((1-y)' * log(1-sigmoid_computed))) + lambda/(2 * m) * sum(theta(2:end) .^ 2);

  % Compute gradient
  % First I computed a list of regularization factors and then remove the First
  % one, effectively reproducing the gradient function :3
  regularized_factor = lambda/m * theta;
  regularized_factor(1) = 0;
  grad = (1/m) * (X' * (sigmoid_computed - y)) + regularized_factor;
  % =============================================================

end
