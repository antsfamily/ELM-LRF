function [ net ] = elmlrfsetup( net, x )
%ELMLRFSETUP Setup ELM-LRF
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

inputmaps = 1;
mapsize = size(squeeze(x(:, :, 1)));

for l = 1 : numel(net.layers)   %  layer
    if strcmp(net.layers{l}.type, 's')
%         mapsize = mapsize / net.layers{l}.scale;
%         assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
        % elm-lrf 下采样(池化)后的特征图与卷积后的大小相同， 无需上述判断
        for j = 1 : inputmaps
            net.layers{l}.f{j} = []; 
        end
    end
    if strcmp(net.layers{l}.type, 'c')
        mapsize = mapsize - net.layers{l}.kernelsize + 1; %d-r+1
%         fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;
        r = net.layers{l}.kernelsize;
        K = net.layers{l}.outputmaps;
        A_init = cell(inputmaps,1);
        for j = 1 : net.layers{l}.outputmaps  %  output map
%             fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;
            for i = 1 : inputmaps  %  input map
%                 net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                A_init{i}(:,j) = reshape(randn(r,r), r^2, 1);% 标准正态分布  %[kernelsize,  kernelsize]  -> [kernelsize*kernelsize,1]
            end
%             net.layers{l}.b{j} = 0; %not need for elm-lrf
        end
        
        % Orthogonalization :SVD 
        for i = 1:inputmaps
            if r*r < K   % r^2 < K
                A = orth(A_init{i}')';
            else
                A = orth(A_init{i});
            end
            % reshape data to r*r
            for j = 1:K
                net.layers{l}.a{i}{j} = reshape(A(:,j), r, r);
            end
        end
        
        inputmaps = net.layers{l}.outputmaps;
        
    end
end


net.BETA = []; % weight beta at the ELM last layer.

end

