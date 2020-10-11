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

% This is embarassing, but the way matrices are represented in Octave make my head
% hurt, meaning that basically I have to do a lot of trial and error to get the right
% orientation of the data

% Create additional ones for each input
hidden_bias = ones(rows(X), 1);

% Add ones to the data
X = [hidden_bias X];

% Compute the activation of the first hidden layer by multiplying
% the X 
hidden_layer_input = X * Theta1';
hidden_layer_act = sigmoid(hidden_layer_input);

% Add bias unit to hidden layer activations
hidden_layer_with_bias = [hidden_bias hidden_layer_act];

% Compute the activation of the output units
last_input = hidden_layer_with_bias * Theta2';
network_output = sigmoid(last_input);

% Generate the target output matrix
sample_output = eye(size(network_output));
sample_output = sample_output(y,:);

% Compute the results for both the positive output and the negative output
positive_error = (-sample_output .* log(network_output));
negative_error = (1 - sample_output) .* log(1 - network_output);
non_regularized_error = (1 / m) * sum(sum(positive_error - negative_error));

% Compute regularization
regularization = (lambda / (2.0*m)) * (sum(sumsq(Theta1(:, 2:end))) + sum(sumsq(Theta2(:, 2:end))));

J = non_regularized_error + regularization;

% -------------------------------------------------------------

% Compute the delta for the thrid layer, which is just the difference between the
% "ground truth" and what we computed
%
% Resulting size: 5000x10 (5000 samples with 10 deltas)
delta_three = network_output - sample_output;
% Then compute the gradient for that layer, which is the delta of the third layer
% multiplied by the activation of the hidden layer
% Resulting size: 10 x 26 (10 outputs for 26 hidden nodes)
% Computation: (10 x 5000) * (5000 x 26) = 10 x 26
Theta2_grad = 1/m * (delta_three' * hidden_layer_with_bias);

% Add gradient regularization
Theta2_grad +=  (lambda / m) * [zeros(rows(Theta2), 1) Theta2(:, 2:end)];

% Compute the delta of the hidden layer by using the provided formula
% Resulting size: 5000 x 26
delta_two = (delta_three * Theta2)(:, 2:end) .* sigmoidGradient(hidden_layer_input);
% Compute the gradient by using X as the activation of the input layer
% Resulting size: 25 x 401
Theta1_grad = 1/m * (delta_two' * X);
Theta1_grad +=  (lambda / m) * [zeros(rows(Theta1), 1) Theta1(:, 2:end)];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
