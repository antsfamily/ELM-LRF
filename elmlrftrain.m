function [ net, er, training_time ] = elmlrftrain( net, x, y, opts )
%ELMLRFTRAIN Train ELM-LRF
%   
% if N <= K*(d-r+1)^2   beta=H'*pinv(I/C+H*H')*T
% if N > K*(d-r+1)^2    beta=pinv(I/C+H'*H)*H'*T
%
%==========================================================================
% Developed based on "cnn" of "DeepLearnToolbox" of rasmusbergpalm on GitHub
%   https://github.com/rasmusbergpalm/DeepLearnToolbox
%   
%==========================================================================
% ---------<LiuZhi>
% ---------<Xidian University>
% ---------<zhiliu.mind@gmail.com>
% ---------<2015/11/24>
%==========================================================================
%

% timing
training_time = cputime;

batchSize = opts.batchsize;

N = size(x, 3); % since x is H-W-N-C, whatever C is
a = fix(N / batchSize); b = rem(N, batchSize);
if b ~= 0, b = 1; end
numBatches = a + b*1;

% K = net.layers{end}
% H = zeros(N, K*(d-r+1)^2);
H = [];

% model
elmlrff = str2func(['@elmlrff_' opts.model]);

for l = 1 : numBatches
    idx = (l-1)*batchSize+1 : min(l*batchSize, N);
    batch_x = x( :, :, idx, : );
    % Compute h :batch
    net = elmlrff(net, batch_x);
    % Combine H
    H = cat(1, H, net.h);

end

clear x batch_x kk idx idxkk;

% Construct T
T = double(y); % nSamples-nClasses
clear y;
% Compute Beta: output weight
if size(H,1) <= size(H,2)  % H is [N, K*(d-r+1)^2]
    net.BETA = H' * ((eye(N,N)/opts.C + H*H') \ T); % A*inv(B)*C  --> A*(B\C)
else
    net.BETA = (eye(size(H,2))/opts.C +H'*H) \ (H'*T); % inv(A)*B  --> A\B
end

[~, label0] = max(T, [], 2);
[~, label] = max(H * net.BETA, [], 2);

bad = find(label0 ~= label);
er = numel(bad) / N;

training_time = cputime - training_time;
end

