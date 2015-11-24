function [ er, bad, testing_time ] = elmlrftest( net, x, y )
%ELMLRFTEST Test ELM-LRF
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
testing_time = cputime;

%forward
net = elmlrff(net, x);

T = net.h * net.BETA; % (N, K(d-r+1)) * (K(d-r+1),nClasses)

[~, label0] = max(y);
[~, label] = max(T,[],2);

bad = find(label0' ~= label);
er = numel(bad) / size(y, 2);  
testing_time = cputime - testing_time;
end

