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
X = [ones(m, 1) X];

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
HX = zeros(m,num_labels);

sample = [1:num_labels];


Y = zeros(m, num_labels);
for (i=1:m)
	Y(i,:) = (y(i) == sample);
end
z2 = X * Theta1';
A2X = sigmoid(z2);
A2X = [ones(m,1) A2X];
disp(size(A2X));
Hx = sigmoid(A2X * Theta2');



theta1sum = sum(sum((Theta1(:, 2:end).^2)));
theta2sum = sum(sum((Theta2(:, 2:end).^2)));
thetasum = theta2sum + theta1sum; 
regularized = (lambda * thetasum)/(2*m);
disp("regularized")
disp(size(regularized))
yHx = -(Y.*log(Hx));
yHx2 = (1 - Y).*(log(1 - Hx));
yHx3 = yHx - yHx;
sumof = (sum(yHx - yHx2)/m);
J = (sum(sumof)) + regularized;


delta_l3 = (Hx - Y);
delta_l2 = (delta_l3 * Theta2).*sigmoidGradient([ones(m,1) z2]);
delta_l2 = delta_l2(:,2:end);

DELTA1 = delta_l2' * X;
DELTA2 = delta_l3' * A2X;

zero1 = zeros(size(Theta1, 1), 1);
zero2 = zeros(size(Theta2, 1), 1);

Theta1_grad = DELTA1./m + [zero1 Theta1(:, 2:end)] * (lambda/m);
Theta2_grad = DELTA2./m + [zero2 Theta2(:, 2:end)] * (lambda/m);















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
