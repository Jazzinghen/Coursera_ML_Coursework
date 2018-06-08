function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% Compute the the entire outups by using matrix multiplication.
% I didn't think that X was a [m, 2] matrix
linear_output = X * theta;

% Compute cost using a single line XD
J = (1/(2*m)) * sum((linear_output - y) .^ 2);

% =========================================================================

end
