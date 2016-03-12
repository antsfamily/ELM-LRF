%demo_elmlrf.m
% A demo of ELM-LRF for MNIST Classiffication
%========================================================================== 
% paper:Huang G, Bai Z, Kasun L, et al. Local Receptive Fields Based 
%   Extreme Learning Machine[J]. Computational Intelligence Magazine IEEE, 
%   2015, 10(2):18 - 29.
%
% myblog:http://blog.csdn.net/enjoyyl/article/details/45724367
%==========================================================================
%
% ---------<Liu Zhi>
% ---------<Xidian University>
% ---------<zhiliu.mind@gmail.com>
% ---------<http://blog.csdn.net/enjoyyl>
% ---------<https://www.linkedin.com/in/%E5%BF%97-%E5%88%98-17b31b91>
% ---------<2015/11/24>
% 

clear all;

%% load MNIST data
data = load('./data/mnist_uint8.mat');
train_x = double(reshape(data.train_x',28,28,60000))/255;
train_y = data.train_y;
train_x = train_x(:,:,1:10000);
train_y = train_y(1:10000, :);
test_x = double(reshape(data.test_x',28,28,10000))/255;
test_y = data.test_y;



%% Setup ELM-LRF
rand('state',0)

elmlrf.layers = {
	struct('type', 'i') %input layer
	struct('type', 'c', 'outputmaps', 10, 'kernelsize', 5) %convolution layer
	struct('type', 's', 'scale', 3) %sub sampling layer
};


opts.batchsize = 500;
opts.model = 'sequential';
% setup
elmlrf = elmlrfsetup(elmlrf, train_x, opts.model);

Cs = [0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
for C = Cs
	opts.C = C;
	%% train ELM-LRF
	[elmlrf, er, training_time] = elmlrftrain(elmlrf, train_x, train_y, opts);
	% disp training error

	fprintf('\nWith C = %f\n-----------------------------------------\nTraining error: %f\nTraining Time:%fs\n', opts.C, er, training_time);

	%% Test ELM-LRF
	% disp testing error
	[er, bad, testing_time] = elmlrftest(elmlrf, test_x, test_y, opts);

	fprintf('\nTesting error: %f\nTesting Time:%fs\n', er, testing_time);

end
