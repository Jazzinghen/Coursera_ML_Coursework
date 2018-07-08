function [X_poly] = polyFeatures(X, p)
  %POLYFEATURES Maps X (1D vector) into the p-th power
  %   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
  %   maps each example into its polynomial features where
  %   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
  %


  % You need to return the following variables correctly.
  X_poly = zeros(numel(X), p);

  % ====================== YOUR CODE HERE ======================
  % Instructions: Given a vector X, return a matrix X_poly where the p-th
  %               column of X contains the values of X to the p-th power.
  %
  %

  % In order to make this a matrix operation I'm creating a matrix of exponents
  % as large as the output, copy the X values to all the locations of the output
  % matrix and then use element-wise power between the twos.
  %
  % It sounds strange to me, so I'll look more into it, but a first search did
  % not return any smarter way to do it. :(
  exponents = repmat((1 : p), numel(X), 1);
  X_poly = repmat(X, 1, p);
  X_poly = X_poly .^ exponents;
  % =========================================================================

end
