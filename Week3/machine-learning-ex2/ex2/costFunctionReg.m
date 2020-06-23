function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


%disp("X\n");
%disp(X);
%disp("y\n");
%disp(y);
disp("Theta \n")
disp(theta);

Hx = sigmoid(X * theta);
%disp("\n")
%disp(-y)
%disp("\n")
%disp(Hx)

yHx = -(y' * log(Hx));
yHx2 = (1 - y') * (log(1 - Hx));
sumof = (sum(yHx - yHx2))/m;

thetasum = sum(theta.^2) - theta(1)^2;
regularized = (lambda * thetasum)/(2*m)


J = sumof + regularized;
disp(m)
d = (Hx - y)' * X;
thetalam = ((lambda)/(m))* theta;
disp(thetalam)
disp(m)
grad = d/m ;
disp(m)
for iter=2:size(thetalam)
	grad(iter) = grad(iter) + thetalam(iter);
end

% =============================================================

end
