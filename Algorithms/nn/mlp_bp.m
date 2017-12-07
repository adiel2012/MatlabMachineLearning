function [W, B] = mlp_bp( x, y, num_iter, num_hidden, rate, momentum )
    [N NA] = size(x);
    [N NC] = size(y);
    X = [x ones(N,1)];
    Amp = 5;
    W = Amp*(rand(NA+1,num_hidden)-1/2);
    B = Amp*(rand(num_hidden+1, NC)-1/2);
    deltaB_previous = zeros(size(B));
    deltaW_previous = zeros(size(W));
    
    for i = 1 : num_iter
        
        H = 1./(1.+exp(-X*W));
        HA = [H  ones(N,1)];
        O = 1./(1.+exp(-HA*B));
        error = sum(sum((y-O).^2))/N

        dEdB = HA'*((O-y).*O.*(1-O));
        dEdW = X'*(((O-y).*O.*(1-O))*B(1:num_hidden,:)'.*(H.*(1-H)));
        
        deltaB = -rate*dEdB + momentum*deltaB_previous; 
        deltaW = -rate*dEdW + momentum*deltaW_previous;
        B = B + deltaB;
        W = W + deltaW;
        
        deltaB_previous = deltaB;
        deltaW_previous = deltaW;
        
    end
end

