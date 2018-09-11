function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% **** FOR-LOOP WAY **** %
% sum = 0;
%
% for i = 1:m,
%     eval = 0;
%
%     % Evaluate hypothesis
%     for j = 1:length(theta),
%         eval = eval + theta(j) * X(i,j);
%     end;
%
%     sum = sum + (eval - y(i)) ^ 2;
% end;
%
% J = sum / (2 * m);

% **** VECTORIZATION WAY **** %
predictions = X * theta;
errors = (predictions - y) .^ 2;

J = sum(errors) / (2 * m);


% =========================================================================

end
