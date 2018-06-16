function plotData(X, y)
  %PLOTDATA Plots the data points X and y into a new figure
  %   PLOTDATA(x,y) plots the data points with + for the positive examples
  %   and o for the negative examples. X is assumed to be a Mx2 matrix.

  % Create New Figure
  figure; hold on;

  % ====================== YOUR CODE HERE ======================
  % Instructions: Plot the positive and negative examples on a
  %               2D plot, using the option 'k+' for the positive
  %               examples and 'ko' for the negative examples.
  %

  % Find the positions of 0s and 1s...
  % This function returns a vector of locations
  found_ones = find(y==1);
  found_zeroes = find(y==0);

  % Plot points based on which category they fall into.
  scatter(X(found_ones, 1), X(found_ones, 2), 'b', 'o');
  scatter(X(found_zeroes, 1), X(found_zeroes, 2), 'r', 'x');
  % =========================================================================

  hold off;

end
