function p = predict(Theta1, Theta2, X)
  %PREDICT Predict the label of an input given a trained neural network
  %   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
  %   trained weights of a neural network (Theta1, Theta2)

  % Useful values
  m = size(X, 1);
  num_labels = size(Theta2, 1);

  % You need to return the following variables correctly
  p = zeros(size(X, 1), 1);

  % ====================== YOUR CODE HERE ======================
  % Instructions: Complete the following code to make predictions using
  %               your learned neural network. You should set p to a
  %               vector containing labels between 1 to num_labels.
  %
  % Hint: The max function might come in useful. In particular, the max
  %       function can also return the index of the max element, for more
  %       information see 'help max'. If your examples are in rows, then, you
  %       can use max(A, [], 2) to obtain the max for each row.
  %

  % I swear I see a pattern here, where I can use only two, swappable,
  % structures. I guess that Thetas come in vectors of Matrices

  % Create the additional ones to append to each row...
  additional_ones = ones(rows(X), 1);

  % Append those ones
  X = [additional_ones X];
  % Compute the new node values, using matrix product and sigmoid
  % computation.
  % This generates the nodes for the hidden layer
  next_nodes = sigmoid(X * Theta1');

  % Do the same thing for the output layer
  next_nodes = [additional_ones next_nodes];
  next_nodes = sigmoid(next_nodes * Theta2');

  % Find the higher probability and return that as result
  [max_prob, p] = max(next_nodes, [], 2);

  % =========================================================================
end
