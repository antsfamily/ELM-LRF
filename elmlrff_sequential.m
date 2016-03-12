function [ net ] = elmlrff_sequential( net, x )
%ELMLRFF_SEQUENTIAL ELM-LRF forward in sequential
%   see './doc/model_sequential.png'
%==========================================================================
% Developed based on "DeepLearnToolbox" of rasmusbergpalm on GitHub
%   https://github.com/rasmusbergpalm/DeepLearnToolbox
%   
%==========================================================================
% ---------<LiuZhi>
% ---------<Xidian University>
% ---------<zhiliu.mind@gmail.com>
% ---------<2015/11/24>
%==========================================================================
%

numLayers = numel(net.layers);
[~,~,~,inputmaps] = size(x);% H-by-W-nImages-by-nChannel
for i = 1:inputmaps
    net.layers{1}.f{i} = x(:,:,:,i);% f: feature
end

for l = 2:numLayers %  for each layer
    if strcmp(net.layers{l}.type, 'c')
        K = net.layers{l}.outputmaps;
        
        for j = 1 : K   %  for each output map
            %  create temp output map
            z = 0;
            for i = 1 : inputmaps   %  for each input map
                %  convolve with corresponding kernel and add to temp output map
                z = z + convn(net.layers{l - 1}.f{i}, net.layers{l}.a{i}{j}, 'valid');
            end
            
            % elm-lrf no bias no activation function
            net.layers{l}.f{j} = z;
        end
        %  set number of input maps to this layers number of outputmaps
        inputmaps = K;
    elseif strcmp(net.layers{l}.type, 's')
        %  downsample
        for j = 1 : inputmaps
            e = fix(net.layers{l}.scale/2);
            % pad 0 and square   compute h
            z = convn(padarray(net.layers{l-1}.f{j}, [e, e]).^2,  ones(net.layers{l}.scale),  'valid');   %  !! replace with variable
            net.layers{l}.f{j} = sqrt(z);
        end
    end
end

%  concatenate all end layer feature maps into vector
net.h = [];
for j = 1 : numel(net.layers{end}.f)
    sa = size(net.layers{end}.f{j});
    net.h = [net.h; reshape(net.layers{end}.f{j}, sa(1) * sa(2), sa(3))];
end
% %  feedforward into output perceptrons
% net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));
net.h = net.h';
end

