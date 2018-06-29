function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
  %NNCOSTFUNCTION Implements the neural network cost function for a two layer
  %neural network which performs classification
  %   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
  %   X, y, lambda) computes the cost and gradient of the neural network. The
  %   parameters for the neural network are "unrolled" into the vector
  %   nn_params and need to be converted back into the weight matrices.
  %
  %   The returned parameter grad should be a "unrolled" vector of the
  %   partial derivatives of the neural network.
  %

  % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
  % for our 2 layer neural network
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1));

  % Setup some useful variables
  m = size(X, 1);

  % You need to return the following variables correctly
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));

  % ====================== YOUR CODE HERE ======================
  % Instructions: You should complete the code by working through the
  %               following parts.
  %
  % Part 1: Feedforward the neural network and return the cost in the
  %         variable J. After implementing Part 1, you can verify that your
  %         cost function computation is correct by verifying the cost
  %         computed in ex4.m
  %
  % Part 2: Implement the backpropagation algorithm to compute the gradients
  %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
  %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
  %         Theta2_grad, respectively. After implementing Part 2, you can check
  %         that your implementation is correct by running checkNNGradients
  %
  %         Note: The vector y passed into the function is a vector of labels
  %               containing values from 1..K. You need to map this vector into a
  %               binary vector of 1's and 0's to be used with the neural network
  %               cost function.
  %
  %         Hint: We recommend implementing backpropagation using a for-loop
  %               over the training examples if you are implementing it for the
  %               first time.
  %
  % Part 3: Implement regularization with the cost function and gradients.
  %
  %         Hint: You can implement this around the code for
  %               backpropagation. That is, you can compute the gradients for
  %               the regularization separately and then add them to Theta1_grad
  %               and Theta2_grad from Part 2.
  %

  % Compute predictions...
  % Create the additional ones to append to each row...
  additional_ones = ones(rows(X), 1);

  % Append those ones
  X = [additional_ones X];
  % Compute the new node values, using matrix product and sigmoid
  % computation.
  % This generates the nodes for the hidden layer
  hidden_layer = sigmoid(X * Theta1');

  % Do the same thing for the output layer
  hidden_layer = [additional_ones hidden_layer];
  prediction = sigmoid(hidden_layer * Theta2');

   % next_nodes is the matrix of outputs

   % Create an identity matrix as big as the output in order to have a results vector...
   y_converter = eye(num_labels);
   vec_output = y_converter(y, :);

   % I am using the same function I used to compute J for the logistic regression, which is the same. The idea is to use
   % vectorialized computations
   J = -(1/m) * sum((vec_output .* log(prediction) + (1-vec_output) .* log(1-prediction))(:));

  % -------------------------------------------------------------

  % =========================================================================

  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
