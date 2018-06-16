function g = sigmoid(z)
  %SIGMOID Compute sigmoid function
  %   g = SIGMOID(z) computes the sigmoid of z.

  % You need to return the following variables correctly
  g = zeros(size(z));

  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the sigmoid of each value of z (z can be a matrix,
  %               vector or scalar).

  % I guess I did a couple idiot things.
  % This function computes the single-value sigmoid per each data point
  % in the provided dataset
  g = 1 ./ (1 + e.^(-z));

  % =============================================================
end
