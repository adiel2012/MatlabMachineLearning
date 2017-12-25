function [ W, bias1, Beta, bias2 ] = initMLP( num_hidden, x,t )
net = feedforwardnet(num_hidden);

net.adaptFcn = 'adaptwb';
net.divideFcn = 'dividerand'; %Set the divide function to dividerand (divide training data randomly).

net.performFcn = 'mse';
%net.trainFcn = 'trainlm'; % set training function to trainlm (Levenberg-Marquardt backpropagation)

%set Layer1
net.layers{1}.name = 'Layer 1';
%net.layers{1}.dimensions = 7;
net.layers{1}.initFcn = 'initnw';
net.layers{1}.transferFcn = 'tansig';

[a b] = size(t);
%set Layer2
net.layers{2}.name = 'Layer 2';
net.layers{2}.dimensions = a;
net.layers{2}.initFcn = 'initnw';
net.layers{2}.transferFcn = 'tansig';

net = configure(net,x,t);
net = init(net) ;

W = net.IW{1,1};
bias1 = net.b{1,1};
Beta = net.LW{2,1};
bias2 = net.b{2,1};

end
