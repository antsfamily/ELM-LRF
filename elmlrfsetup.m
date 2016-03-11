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

[mapsize(1),mapsize(2),~,inputmaps] = size(x);% H-by-W-by-nChannel-by-nImages
% mapsize = size(squeeze(x(:, :, 1)));

for l = 1 : numel(net.layers)   %  layer
    if strcmp(net.layers{l}.type, 's')
%         mapsize = mapsize / net.layers{l}.scale;
%         assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
        % elm-lrf 
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
        for j = 1 : K  %  output map
%             fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;
            for i = 1 : inputmaps  %  input map
%                 net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                A_init{i}(:,j) = reshape(randn(r,r), r^2, 1); %[kernelsize,  kernelsize]  -> [kernelsize*kernelsize,1]
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
                net.layers{l}.a{i}{j} = reshape(A(:,j), r, r);% a is the weight with kernelsize r*r
            end
        end
        
%         inputmaps = net.layers{l}.outputmaps;
        inputmaps = inputmaps*net.layers{l}.outputmaps;
        
    end
end


net.BETA = []; % weight beta at the ELM last layer.

end

