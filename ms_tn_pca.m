%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Multi-graph TN-PCA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Inputs:
% X_cell - a cell of 3-tensors
% K - number of factors to extract
%
% options - a struct with possible values:
% options.proj - whether or not the v_k should be orthogonal (1=yes, 0=no)
% options.maxit - max iteration
% options.thr - threshold to establish convergence
% options.penalty - choice between "L1" or "L2" objective function
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Returns:
% V - a cell of p_i*K matrices (network modes)
% D - a cell of K length vectors 
% U - a N*K matrix of subject modes
% X hat - a cell of residual tensors
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[V,D,U,Xhat_cell] = ms_tn_pca(X_cell,K,options)

narginchk(2, 3);

% default parameters
penalty = "L2";
proj = 1;
maxit = 1e4; thr = 1e-5;
Local_Search_Iter = 5;
localsiter = 0;

% set up cells to store results
U = []; V = {}; D = {};
Xhat_cell = X_cell;

% constants for the loop
tmp_dim = size(X_cell{1});
N = tmp_dim(3);
r = length(X_cell);

if isfield(options,'proj')
    proj = options.proj;
end

if isfield(options,'penalty')
    penalty = options.penalty;
end

for i=1:r
    tmp_dim = size(X_cell{i});
    scales(i) = tmp_dim(2);

    V{i} = [];
    D{i} = [];
end    


for h=1:K

    localsiter = 0;
    
    while(Local_Search_Iter>localsiter)
        localsiter = localsiter +1;

        % common scores, so we only need to initialize one value of u
        u = randn(N,1);
        u = u/norm(u);

        for i=1:r
            % initialize a v for each scale
            current_v{i} = randn(scales(i),1);
            current_v{i} = current_v{i}/norm(current_v{i});
        end    

        for i=1:r
            P{i} = eye(scales(i));
            
            if h>1 && proj==1
                % only compute the projections if we want v_h orthogonal
                P{i} = eye(scales(i)) - V{i}*V{i}';
            end
                
        end    

        ind = 1;
        iter = 1;
        objective = 0;

        for i=1:r
            objective = objective + (P{i}*current_v{i})'*double(ttv(Xhat_cell{i},P{i}*current_v{i},2))*u;  
        end

        while ind>thr && maxit>iter
            last_objective = objective(end);
            
            if penalty=="L1"
                % find the L1 penalty if specified
                uhat = zeros(N,1);
                
                for i=1:r
                    uhat = uhat + double(ttv(Xhat_cell{i},current_v{i},1))'*current_v{i};
                end
                
            else
                % compute the L2 penalty if the user didn't specify L1
                M = zeros(N,N);
                
                for i=1:r
                    % compute the matrix we need to find u
                    alpha = double(ttv(Xhat_cell{i},current_v{i},1))'*current_v{i};
                    M = M + alpha*alpha';
                end
                
                [uhat,tmp] = eigs(M, 1);
            end

            u = uhat/norm(uhat);

            for i=1:r
                % update each v individually conditioned on u
                [v_tmp,tmp] = eigs(P{i}*double(ttv(Xhat_cell{i},u,3))*P{i},1);
                current_v{i} = v_tmp;
            end    

            objective_tmp = 0;
            for i=1:r
                % now compute the new objective
                objective_tmp = objective_tmp + (P{i}*current_v{i})'*double(ttv(Xhat_cell{i},P{i}*current_v{i},2))*u;
            end    

            % update the loop parameters
            objective = [objective, objective_tmp];
            ind = abs((objective(end) - last_objective)/objective(1));
            iter = iter + 1;
        end
        
        Obj_localsearchiter(localsiter) = 0;
        for i=1:r
            % compute d
            % update the objective for this local search iteration
            current_d{i} = current_v{i}'*double(ttv(Xhat_cell{i},u,3))*current_v{i};
            tmp_Xhat{i} = Xhat_cell{i} - full(ktensor(current_d{i},current_v{i},current_v{i},u));
            Obj_localsearchiter(localsiter) = Obj_localsearchiter(localsiter) + current_v{i}'*double(ttv(Xhat_cell{i},current_v{i},2))*u;
        end  
        
        if localsiter == 1
            % if this is our first search then we don't have anything
            % better
            for i=1:r
                best_v{i} = current_v{i};
                best_d{i} = current_d{i};
            end    
            
            best_u = u;
        else
            % otherwise, see if we've beaten our previous best
            best_obj = max(Obj_localsearchiter);
            
            if Obj_localsearchiter(localsiter) > best_obj
                for i=1:r
                    best_v{i} = current_v{i};
                    best_d{i} = current_d{i};
                end
                
                best_u = u;
            end    
        end    
    end         
    
    for i=1:r
        % deflate; save the best parameters
        Xhat_cell{i} = Xhat_cell{i} - full(ktensor(best_d{i},best_v{i},best_v{i},best_u));
        V{i} = [V{i}, best_v{i}];
        D{i} = [D{i}, best_d{i}];
    end    
    
    U = [U, best_u];
end    


