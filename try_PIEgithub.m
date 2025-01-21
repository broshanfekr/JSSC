clc;
clear;

addpath('L1_ADMM/');
addpath('toolbox/');
addpath('JSSC/');

addpath('measure/');
addpath('data/');

%% load dataset
% This section assumes that data are stored in variables X and Y
% X is a data matrix with size of n*d
% Y is true labels with size n*1

load('data/PIE_N1340_D1024.mat');
dataset_name = "fashionmnist";
% data_path = strcat(dataset_name, ".mat");
% load(data_path)
% 
% FEA = zeros(size(fea, 1), size(fea, 2)*size(fea, 3));
% for i = 1:size(fea, 1)
%    tmp_img = fea(i, :, :, :);
%    tmp_img = squeeze(tmp_img);
%    tmp_img = rgb2gray(tmp_img);
%    tmp_img = reshape(double(tmp_img), 1, size(fea, 2)*size(fea, 3));
%    FEA(i, :) = tmp_img(1,:);
% end
% fea = FEA;
% 
% Y = double(gnd);
% X = X;

%% Data processing
dataSetName = 'PIE';
nExperiment = 1;
data_num = size(X, 1);
class_num = length(unique(Y));
label = Y;
nCluster = length(unique(label));



K=[4, 5];
beta=[0.025];
lambda = [0.1, 1];
alph = [0.0001];
rho =[0.01];
corruption = 0;%add noise
corruption1 = [0];
maxiter=linspace(1,6,6);
%maxiter=5;

results_JSSCSSC=zeros(nExperiment, 5);

breakpoint_path = strcat('JSSC_',dataset_name,'_clip.mat');
if exist(breakpoint_path)
    load(breakpoint_path);
else
    validation_nmi = zeros(length(K), length(beta), length(lambda), length(alph), length(rho), length(maxiter));
    resutls = [];
end

for cc=1:length(corruption1)
    corruption1_cc = corruption1(cc);
    for kk=1:length(K)
        K_kk = K(kk);
        for bb=1:length(beta)
            beta_bb=beta(bb);
            for ll=1:length(lambda)
                lambda_ll=lambda(ll);
                for aa=1:length(alph)
                    alph_aa=alph(aa);
                    for rr=1:length(rho)
                        rho_rr=rho(rr);
                        for ii=1:length(maxiter)
                            maxiter_ii=maxiter(ii);
                            
                            if validation_nmi(kk, bb, ll, aa, rr, ii) ~= 0
                                disp(['K: ',num2str(K_kk,'%d'), '   beta: ',num2str(beta_bb,'%01.3f'), '   lambda: ',num2str(lambda_ll,'%01.3f'), '   alph: ',num2str(alph_aa,'%01.3f'), '   rho: ',num2str(rho_rr,'%01.3f'), '   maxiter: ',num2str(maxiter_ii,'%d'),'   already done.']);
                                continue;
                            end

                            for iExperiment = 1:nExperiment
                                %add noise
                                [D,N] = size(X);
                                a=zeros(size(X));
                                corruption_mask = randperm( D*N, round( corruption*D*N ) );
                                a(corruption_mask)=X(corruption_mask);
                                %a = imnoise(a,'speckle',0.01);
                                %a = imnoise(a,'salt & pepper',0.01);
                                %a = imnoise(a,'poisson');%≤¥À…‘Î
                                a=imnoise(a,'gaussian',0,0.01);
                                X(corruption_mask)=a(corruption_mask);
                                corruption_mask1=randperm(N);
                                b=fix(corruption1_cc*N);
                                X(:,corruption_mask1(1:b))=zeros(D,b);

                                X = NormalizeFea(X, 1);    %%% Normalization

                                s=label;
                                %Z=X'\X';   % commented by bero
                                %figure(1)  % commented by bero

                                %colorbar   % commented by bero
                                %map = [1 1 1           % commented by bero
                                %    1 0.94118 0.96078  % commented by bero
                                %    1 0.89412 0.88235  % commented by bero
                                %    1 0.62745 0.47843  % commented by bero
                                %    1 0.49804 0.31373  % commented by bero
                                %    1 0.27059 0];   % commented by bero
                                %clims = [0 0.5];    % commented by bero
                                %A=imagesc(abs(Z)',clims);   % commented by bero

                                %A=imagesc(abs(Z));
                                %colorbar     % commented by bero
                                %colormap(map)   % commented by bero
                                %JSSC
                                tol = 1e-6;
                                normalizeColumn = @(data) cnormalize_inplace(data);

                                %JSSCSSC
                                %tol = 1e-6;
                                %normalizeColumn = @(data) cnormalize_inplace(data);
                                % [B,E,C_JSSC,time_JSSC] = JSSC_PAMnoalpha(X',beta_bb,lambda_ll,0,rho_rr,K_kk,tol,maxiter);
                                [BSSC,ESSC,C_JSSCSSC,time_JSSCSSC] = JSSC_PAMcutSSC(X',beta_bb,lambda_ll,0,rho_rr,K_kk,tol,maxiter_ii);
                                %C_JSSC = C_JSSC(1:N,:);
                                C_JSSCSSC = C_JSSCSSC(1:size(X,1),:);
                                W_JSSCSSC = abs(C_JSSCSSC) + abs(C_JSSCSSC');   %%%  affinity matrix
                                Y_JSSCSSC = SpectralClustering1(W_JSSCSSC, nCluster);
                                %EvaluationJSSC
                                accr_JSSCSSC  = evalAccuracy(s, Y_JSSCSSC);
                                nmi_JSSCSSC  = nmi(Y_JSSCSSC, s);
                                
                                [metrics.ca,metrics.nmi,metrics.ar,metrics.f1,~,~] = compute_metrics(s, Y_JSSCSSC);

                                dataformat_JSSCSSC = '%d-th experiment:  accr_JSSC = %f, nmi_JSSC = %f, time=%f\n';
                                dataValue_JSSCSSC = [metrics.ca, metrics.nmi, metrics.ar, metrics.f1, time_JSSCSSC];
                                results_JSSCSSC(iExperiment, :)=dataValue_JSSCSSC;
                            end

                            % output
                            dataValue_JSSCSSC=mean(results_JSSCSSC, 1);
                            fprintf('\nAverage:iter=%d, beta=%d, lambda=%d, alpha=%d, rho=%d, K=%d:  acc = %f, nmi = %f, ari = %f, f1_score = %f, time=%f\n', maxiter(ii), beta(bb), lambda(ll), alph(aa), rho(rr), K(kk), dataValue_JSSCSSC(:));

                            validation_nmi(kk, bb, ll, aa, rr, ii) = metrics.nmi;
                            resutls = [resutls; [K_kk, beta_bb, lambda_ll, alph_aa, rho_rr, maxiter_ii, dataValue_JSSCSSC]];
                            save(breakpoint_path, 'validation_nmi', "resutls");

                        end
                    end
                end
            end
        end
    end
end