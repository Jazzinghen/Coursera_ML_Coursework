function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  %GRADIENTDESCENT Performs gradient descent to learn theta
  %   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
  %   taking num_iters gradient steps with learning rate alpha

  % Initialize some useful values
  m = length(y); % number of training examples
  J_history = zeros(num_iters, 1);

  for iter = 1:num_iters

      % ====================== YOUR CODE HERE ======================
      % Instructions: Perform a single gradient step on the parameter vector
      %               theta.
      %
      % Hint: While debugging, it can be useful to print out the values
      %       of the cost function (computeCost) and gradient here.
      %
      linear_output = X * theta;

      % The formula is somewhere along the lines of "For each tetha
      % compute the difference with the cost function"
      %
      % This means that we can use theta (1 x n) and multiply it by
      % the result of all the Xs transposed (n x m) multiplied by
      % the distance of the linear output from the real output (m x 1)
      %
      % (X' * (linear_output - y)) computes sum(h(x^i) - y) for each
      % tetha, allowing then to just run a scalar multiplication and a
      % vector subtraction
      next_theta = theta - (alpha/m) * (X' * (linear_output - y));

      theta = next_theta;
      % ============================================================

      % Save the cost J in every iteration
      J_history(iter) = computeCost(X, y, theta);

  end

end
