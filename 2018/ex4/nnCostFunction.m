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
  first_activation = X * Theta1';
  hidden_layer = sigmoid(first_activation);

  % Do the same thing for the output layer
  hidden_layer = [additional_ones hidden_layer];
  second_activation = hidden_layer * Theta2';
  prediction = sigmoid(second_activation);

  % next_nodes is the matrix of outputs

  % Create an identity matrix as big as the output in order to have a results vector...
  y_converter = eye(num_labels);
  vec_output = y_converter(y, :);

  % I am using the same function I used to compute J for the logistic regression, which is the same. The idea is to use
  % vectorialized computations
  positive_cost_component = vec_output .* log(prediction);
  negative_cost_component = (1-vec_output) .* log(1-prediction);

  first_hidden_regularization = sum(Theta1(:, 2:end)(:) .^ 2);
  second_hidden_regularization = sum(Theta2(:, 2:end)(:) .^ 2);

  % Compute the cost using the various components. Did it like this because the command was too long. And it still
  % counts as too long, in my opinion.
  %
  % I had to use the (:) command in order to transform the matrices in vectors otherwise the sum function will compute
  % the sum along one of the dimensions
  J = -(1/m) * sum((positive_cost_component + negative_cost_component)(:)) + (lambda/(2*m) * (first_hidden_regularization + second_hidden_regularization));

  % -------------------------------------------------------------
  % Compute gradients

  % Compute the first difference. This is basically how far the prediction was with respect to the ground truth.
  diff_3 = prediction - vec_output;

  % Compute second layer differences and gradient
  %
  % First step is to... Mh. Compute the difference between the activation value of the hidden layer with the activation
  % it should have had if we returned the ground truth.
  % To do this we perform a backward computation using the delta of the output layer and compute the sum of each delta
  % multiplied by the weights that contributed to that delta.
  % This will results in a matrix that has a number of rows equal to the examples and a number of columns equal to each
  % node in the hidden layer.
  %
  % After that we compute the partial derivatives for each weight
  diff_2 = (diff_3 * Theta2)(:, 2:end) .* sigmoidGradient(first_activation);
  Theta2_grad = 1/m * (diff_3' * hidden_layer);

  % Compute regularization factors and add them to the output gradient
  Theta2_reg = (lambda/m) * Theta2;
  Theta2_reg(:, 1) = 0;
  Theta2_grad = Theta2_grad + Theta2_reg;

  % Compute the first layer gradient
  %
  % This step is equal to the other but we don't need to compute the deltas as the values in the layer are the input
  % values, which cannot be changed.
  Theta1_grad = 1/m * (diff_2' * X);
  Theta1_reg = (lambda/m) * Theta1;
  Theta1_reg(:, 1) = 0;
  Theta1_grad = Theta1_grad + Theta1_reg;

  % =========================================================================

  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
