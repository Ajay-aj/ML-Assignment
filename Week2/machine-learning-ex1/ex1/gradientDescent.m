function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
Xval = X(:,2);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    iter
    hx1 = X * theta;
    diff1 = sum(hx1 - y);
    diff2 = sum((hx1 - y)' * Xval);
    total1 = (alpha * diff1)/m;
    total2 = (alpha * diff2)/m;
    temp1 = theta(1) - total1;
    temp2 = theta(2) - total2;
    theta(1) = temp1;
    theta(2) = temp2;
    %disp("Cost is = "), disp(J_history(iter))

end

end
