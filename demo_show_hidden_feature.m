
%% Step0 Set an Breakpoint at the end of "elmlrff.m", then run "demo_elmlrf_NORB.m" to the breakpoint

%% Step1 Copy following codes in MATLAB's Command Window, press "Enter"
figure(1)
displayGrayNetwork(reshape(x(:,:,1:100,2),32*32,100))
title('Origional images')

figure(2)
displayGrayNetwork(reshape(net.layers{2,1}.f{4}(:,:,1:100),29*29,100))
title('Feature map after convolution')

figure(3)
displayGrayNetwork(reshape(net.layers{3,1}.f{4}(:,:,1:100),29*29,100))
title('Feature map after pooling')
