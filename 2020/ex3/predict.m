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

  % Create the additional ones to append to each row...
  additional_ones = ones(rows(X), 1);

  % Add ones to the X data matrix, as with predictOneVsAll
  X = [additional_ones X];
  % Compute the activations for the nodes in the first hidden layer
  second_layer = sigmoid(X * Theta1');
  
  second_layer = [additional_ones second_layer];
  % Compute the activations for the output layer for each input
  third_layer = sigmoid(second_layer * Theta2');
  
  % Get the "Activated" output node
  [tmp,p] = max(third_layer, [], 2);

% =========================================================================


end
