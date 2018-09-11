function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

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
    predictions = X * theta;
    errors = predictions - y;
    sum0 = sum(errors) / m;
    sum1 = sum(errors .* X(:, 2)) / m;

    % Update thetas
    theta(1) = theta(1) - alpha * sum0;
    theta(2) = theta(2) - alpha * sum1;
    J_history(iter);

    % FOR-LOOP WAY, SLOW AF
    % % Calculate theta 1
    % sum = 0;
    % for i = 1:m,
    %     eval = 0;
    %
    %     % Evaluate hypothesis
    %     for j = 1:length(theta),
    %         eval = eval + theta(j) * X(i,j);
    %     end;
    %
    %     sum = sum + (eval - y(i));
    % end;
    %
    % % Calculate theta 2
    % sum2 = 0;
    % for i = 1:m,
    %     eval = 0;
    %
    %     % Evaluate hypothesis
    %     for j = 1:length(theta),
    %         eval = eval + theta(j) * X(i,j);
    %     end;
    %
    %     sum2 = sum2 + (eval - y(i)) * X(i, 2);
    % end;
    %
    % theta(1) = theta(1) - alpha * (sum/m);
    % theta(2) = theta(2) - alpha * (sum2/m);
    %
    % J_history(iter);

end

end
