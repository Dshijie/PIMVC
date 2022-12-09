function [Y,obj] = PIMVC(X,inv_GS,linshi_LS,St2,St3,G,ind_folds,lambda,beta,max_iter,dim)
% ------ ³õÊ¼»¯ -------- %
rand('seed',6666);
normX = 0;
for iv = 1:length(X)
    options = [];
    options.ReducedDim = dim;
    [P1,~] = PCA1(X{iv}', options);
    Piv{iv} = P1';
    XG{iv} = X{iv}*G{iv}';
    normX = normX+norm(X{iv},'fro')^2;
end
numInst  = size(G{1},1); 
Y = rand(dim,numInst);

for iter = 1:max_iter
    % ------------- Y ------------- %
    linshi_H1 = 0;
    for iv = 1:length(X)
        linshi_H1 = linshi_H1 + Piv{iv}*XG{iv};
    end
    Y = linshi_H1*inv_GS;
    
    % ----------------- Piv ------------------- %
    for iv = 1:length(X)
        linshi_M = St3{iv}*Y';
        linshi_M(isnan(linshi_M)) = 0;
        linshi_M(isinf(linshi_M)) = 0;
        [linshi_U,~,linshi_V] = svd(linshi_M','econ');
        linshi_U(isnan(linshi_U)) = 0;
        linshi_U(isinf(linshi_U)) = 0;
        linshi_V(isnan(linshi_V)) = 0;
        linshi_V(isinf(linshi_V)) = 0;        
        Piv{iv} = linshi_U*linshi_V'*St2{iv};
    end

    % -------------- obj --------------- %
    linshi_obj = 0;
    for iv = 1:length(X)
        linshi_obj = linshi_obj+norm(Piv{iv}*X{iv}-Y*G{iv},'fro')^2+lambda*norm(Piv{iv},'fro')^2;        
    end
    obj(iter) = (linshi_obj+beta*trace(Y*linshi_LS*Y'))/normX;
    if iter >3 && abs(obj(iter)-obj(iter-1))<1e-5
        %iter
        break;
    end
end
end