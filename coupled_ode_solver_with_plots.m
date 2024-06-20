% Define model parameters

N = 50; % N - Number of agents

T = 2; % T - Number of topics

K = 1; % K - The connectivity

alpha = 0.05; % alpha - the controversialness


phi = eye(T) + rand(T) * 2 - 1;
phi(logical(eye(size(phi)))) = 1; 

A = randi([0, 1], N, N);

mu = zeros(T, 1); 

sigma = sqrt(2.5); 

x0 = mvnrnd(mu, sigma^2 * eye(T), N)'; 

ode_func = @(t, x) myODEfunc(t, x, K, alpha, phi, A, N, T);

[t, X] = ode45(ode_func, [0, 50], x0(:));
X = reshape(X, [length(t), T, N]);

figure;
subplot(2, 1, 1);
hold on;
for i = 1:N
    plot(t, X(:, 1, i), 'Color', rand(1,3));
end

xlabel('Time');
ylabel('Component 1');
title('Variation of First opinion');


subplot(2, 1, 2);
hold on;
for i = 1:N
    plot(t, X(:, 2, i), 'Color', rand(1,3));
end

xlabel('Time');
ylabel('Component 2');
title('Variation Second opinion');

function dxdt = myODEfunc(~, x, K, alpha, phi, A, N, T)
    dxdt = zeros(N*T, 1);
    x = reshape(x, [T, N]);
    for i = 1:N
        sum_term = 0;
        for j = 1:N
            sum_term = sum_term + A(i, j) * tanh(alpha * phi * x(:, j));
        end
        dxdt((i-1)*T + 1:i*T) = -x(:, i) + K * sum_term;
    end
    dxdt = dxdt(:);

end