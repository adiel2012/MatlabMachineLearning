init
%  num_hidden, rate, momentum
iters = [100, 500, 1000, 2000, 5000];
hidden_neurons = [2, 5, 10 , 15, 20];
rates = [0.01, 0.05, 0.1, 0.2, 0.3];
momentums = [0.1, 0.2, 0.5, 0.8];
N = size(x_test,1);

for i = 1 : size(iters,2)
    for j = 1 : size(hidden_neurons,2)
        for k = 1 : size(rates,2)
            for m = 1 : size(momentums,2)
               [W, B, error_train] = mlp_bp(x_train, y_train, iters(i), hidden_neurons(j), rates(k), momentums(m)); 
               error_test = sum(sum((y_test-(1./(1.+exp(-([(1./(1.+exp(-([x_test ones(N,1)])*W))) ones(N,1)])*B)))).^2))/N;
               disp(sprintf('iterations: %d, hidden neurons: %d, rate: %d, momentum: %d --> train error: %d    test error: %d',iters(i), hidden_neurons(j), rates(k), momentums(m), error_train, error_test))
            end
        end
    end
end

%mlp_bp(x, y, 10000000, 2, 0.1, 0.3);