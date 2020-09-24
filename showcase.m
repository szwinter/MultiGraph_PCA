%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% minimal working example to highlight the algorithm
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
rng('default')

% need ktensor, full, and ttv
addpath('tensor_toolbox') 
addpath('tensor_toolbox/met')

% simualate a tensor of dim 50x50x100 and another of 200x200x100,
% both with true rank 10
N = 100;
P = [50 200];
K_true = 10;

for h=1:2
    D{h} = sort(rand(1, 10), 'descend');
    
    tmp = randn(P(h), N);
    [V{h},~] = eigs(tmp*tmp', K_true);
    
    % shift the latent factors for one of the groups
    tmp = rand(N, K_true);
    tmp(1:(N/2),:) = tmp(1:(N/2),:) + 1;
    U{h} = normc(tmp);

    % multiply by a constant to get reasonable numbers
    % note - ktensor(D', V, V, U) is also the way to reconstruct 
    % a low rank approximation from the estimate componenets 
    X{h} = full(10^8*ktensor(D{h}', V{h}, V{h}, U{h}));
end

% compute a rank 2 approximation with the L2 penalty
K_recon = 2;
options = struct('proj',1);
options.penalty = 'L2';
[Vk, Dk, Uk, residuals] = ms_tn_pca(X, K_recon, options);

% plot the first 2 latent factors, colored by group
colormap flag;
scatter(Uk(:,1),Uk(:,2), [],[zeros(1, N/2) ones(1, N/2)],'filled');
xlabel('u_1');
ylabel('u_2');

