clc;clear;
addpath('./dataset/');
addpath('./fun');
data={'bbcsport4vbigRnSp','handwritten-5view','Caltech101-7','animal'};
Dataname=data{1};
delset=[0.1,0.3,0.5,0.7];
lambdaset=10.^[-5:1:5];
betaset=10.^[-5:1:5];
kset=3:2:15;
options=[];
options.NeighborMode='KNN';
options.WeightMode='Binary';      % Binary  HeatKernel
max_iter = 100;
for del=0.5
    Datafold=[Dataname,'_del_',num2str(del),'.mat'];
    load(Dataname);         % 一列一个样本
    load(Datafold);
    numClust=length(unique(Y));
    numSample=length(Y);
    numView=length(X);
    for k=13
        options.k=k;
        for lambda=10
        for beta=0.1
        for f= 1:5
            t1=clock;

            fold = folds{f};
            linshi_GG = 0;
            linshi_LS = 0;
            for iv = 1:length(X)
                %X1{iv} = X{iv};
                X1{iv} = NormalizeFea(X{iv},0);
                ind_1 = find(fold(:,iv) == 1);
                ind_0 = find(fold(:,iv) == 0);
                X1{iv}(:,ind_0) = [];
                % ------------- 构造缺失视角的索引矩阵 ----------- %
                linshi_W = diag(fold(:,iv));
                linshi_W(:,ind_0) = [];
                G{iv} = linshi_W;
                linshi_St = X1{iv}*X1{iv}'+lambda*eye(size(X1{iv},1));
                St2{iv} = mpower(linshi_St,-0.5);
                St3{iv} = St2{iv}*X1{iv}*G{iv}';
                linshi_GG = linshi_GG+fold(:,iv);
                linshi_W = full(constructW(X1{iv}',options));
                W_graph = (linshi_W+linshi_W')*0.5;
                W_graph = G{iv}*W_graph*G{iv}';
                Sum_S = sum(W_graph);
                linshi_LS = linshi_LS+diag(Sum_S)-W_graph;
            end
            inv_GS = inv(linshi_LS*beta+diag(linshi_GG));
            [P,obj] = PIMVC(X1,inv_GS,linshi_LS,St2,St3,G,fold,lambda,beta,max_iter,numClust);
            new_F = P';
            pre_labels=kmeans(real(new_F),numClust,'emptyaction','singleton','replicates',20,'display','off');
            res(f,:)=ClusteringMeasure(Y, pre_labels)*100;

            t2=clock;
            t(f)=etime(t2,t1)
        end
        mean_acc=mean(res(:,1));
        std_acc = std(res(:,1));
        mean_nmi=mean(res(:,2));
        std_nmi = std(res(:,2));
        mean_pur=mean(res(:,3));
        std_pur = std(res(:,3));   
        disp(['acc=',num2str(mean_acc),'----nmi=',num2str(mean_nmi),'----pur=',num2str(mean_pur)]);
        end
        end
    end
end
