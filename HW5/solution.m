%% Clear workspace and close windows
    
    clc
    clear
    close all
    
%% Feature Computation for 8x8 block in the image cheetah.bmp, compute the feature X.

    block_size = 8;
    ZigZagPattern = table2array(readtable('Zig-Zag Pattern.txt'))+1;
    
    image = im2double(imread('cheetah.bmp'));
    [height,width] = size(image);
    image = padarray(image,[8,8],'symmetric');
    ground_truth = im2double(imread('cheetah_mask.bmp'));
    dctZigZag = zeros(height,width,block_size*block_size);
    
    % Iterating through the image in 8x8 blocks through a sliding window
    % Compute feature vectors for test image
    for h = 1:height
        for w = 1:width
            dctBlock = dct2(image(h+4:h+11,w+4:w+11));
            % Rearranging the DCT Components in ZigZag Patterned Vector
            for i = 1:block_size
                for j = 1:block_size
                    dctZigZag(h,w,ZigZagPattern(i,j)) = dctBlock(i,j);
                end
            end
        end
    end
    
    % Load training dataset
    DCT = load('TrainingSamplesDCT_8.mat');
    Train_FG = DCT.TrainsampleDCT_FG;
    Train_BG = DCT.TrainsampleDCT_BG;
    
    % Assuming prior distribution is (# of samples of A)/total training samples
    N_BG = size(Train_BG,1);
    N_FG = size(Train_FG,1);
    
    % Based on dataset size
    prior_BG = N_BG/(N_BG+N_FG);
    prior_FG = N_FG/(N_BG+N_FG);    

% % %% Generating the output image for the two features
    C = 8;
    epsilon = 1e-8;
    n_iter = 100;
    
    pi_fg = cell(5,1);
    pi_bg = cell(5,1);
    mu_fg= cell(5,1);
    mu_bg = cell(5,1);
    cov_fg = cell(5,1);
    cov_bg= cell(5,1);
    
    % Create 5 different mixtures based on different initializations
    for i =1:5
        [pi_fg{i},mu_fg{i},cov_fg{i}] = compute_param(C,epsilon,n_iter,Train_FG);
        [pi_bg{i},mu_bg{i},cov_bg{i}] = compute_param(C,epsilon,n_iter,Train_BG);
    end

    dim = [1,2,4,8,16,24,32,40,48,56,64];
    [~,dimlen] = size(dim);
    error = zeros(25,dimlen);

    for i = 1:5 %% BG
        figure
        for j = 1:5 %% FG
            for k = 1:dimlen
                error(5*(i-1)+j,k) = prediction(dctZigZag,height,width,dim(k),C,pi_bg{i},pi_fg{j},mu_bg{i},mu_fg{j},cov_bg{i},cov_fg{j},prior_BG,prior_FG,ground_truth);
            end
            plot(dim,error(5*(i-1)+j,:),'*-')
            hold on
        end
        grid on
        legend('FG mixture 1','FG mixture 2','FG mixture 3','FG mixture 4','FG mixture 5')
        title(['PoE vs # of features - BG mixture ',num2str(i)])
    end
    
    
%% Generating the output image for the two features
    epsilon = 1e-8;
    n_iter = 100;
   
    C = [1,2,4,8,16,32];
    [~,complen] = size(C);
    dim = [1,2,4,8,16,24,32,40,48,56,64];
    [~,dimlen] = size(dim);
    error_b = zeros(complen,dimlen);
    
    pi_fg = cell(complen,1);
    pi_bg = cell(complen,1);
    mu_fg= cell(complen,1);
    mu_bg = cell(complen,1);
    cov_fg = cell(complen,1);
    cov_bg= cell(complen,1);

        % Create 5 different mixtures based on different initializations
    for i =1:complen
        [pi_fg{i},mu_fg{i},cov_fg{i}] = compute_param(C(i),epsilon,n_iter,Train_FG);
        [pi_bg{i},mu_bg{i},cov_bg{i}] = compute_param(C(i),epsilon,n_iter,Train_BG);
    end
    
    figure
    for i = 1:complen
        for k = 1:dimlen
            error_b(i,k) = prediction(dctZigZag,height,width,dim(k),C(i),pi_bg{i},pi_fg{i},mu_bg{i},mu_fg{i},cov_bg{i},cov_fg{i},prior_BG,prior_FG,ground_truth);
        end
        plot(dim,error_b(i,:),'o-')
        hold on
    end
    grid on
    legend('C=1','C=2','C=4','C=8','C=16','C=32');
    title('PoE vs feature dimension for different C');
    
        
%% UTILITY FUNCTIONS
    
function [error] = prediction(dctZigZag,height,width,dim,C,pi_bg,pi_fg,mu_bg,mu_fg,cov_bg,cov_fg,prior_BG,prior_FG,ground_truth)

    g_grass = zeros(height*width,1);
    g_cheetah = zeros(height*width,1);
    
    dct_dimvec = reshape(dctZigZag(:,:,1:dim),[],dim);
    mu_fg_dim = mu_fg(:,1:dim);
    mu_bg_dim = mu_bg(:,1:dim);
    cov_fg_dim = cov_fg(:,1:dim,1:dim);
    cov_bg_dim = cov_bg(:,1:dim,1:dim);
    
    for j=1:C
        g_grass = g_grass + mvnpdf(dct_dimvec,mu_fg_dim(j,:),squeeze(cov_fg_dim(j,:,:)))*pi_fg(j);
        g_cheetah = g_cheetah + mvnpdf(dct_dimvec,mu_bg_dim(j,:),squeeze(cov_bg_dim(j,:,:)))*pi_bg(j);
    end
    
    X = prior_BG*g_grass > prior_FG*g_cheetah;
    X = reshape(X,height,width);
    
    error = error_computation(ground_truth,X,prior_FG,prior_BG); 
end


function [pi,mu,cov] = compute_param(C,threshold,n_iter,TrainData)
    
    % Initialize mixture model parameters
        pi = (randi(1000,C,1)-1)/1000;
        pi = pi/sum(pi);
        mu = randn(C,64);
        cov = zeros(C,64,64);
        
        N_train = size(TrainData,1);
            
        for i=1:C
            cov(i,:,:) = diag(randn(64,1).^2 + 1);
        end

        for step=1:n_iter
            h_ij = zeros(N_train,C);
            for j=1:C
                h_ij(:,j) = mvnpdf(TrainData,mu(j,:),squeeze(cov(j,:,:)))*pi(j);
            end
            
            h_ij = h_ij./(sum(h_ij,2));
            
            for j = 1:C
                diagonal = sum(h_ij(:,j).*((TrainData-mu(j,:)).^2),1)./sum(h_ij(:,j),1) + threshold;
                cov(j,:,:) = diag(diagonal);
            end
            mu = (h_ij'*TrainData)./sum(h_ij,1)';
            pi = sum(h_ij,1)./N_train;
        end
end


function [probability_error] = error_computation(ground_truth,prediction,FG_prior,BG_prior)

    probability_error_cheetah = sum(ground_truth & ~prediction,'all')/sum(ground_truth,'all');
    probability_error_grass = sum(~ground_truth & prediction,'all')/sum(~ground_truth,'all');
    probability_error = (FG_prior*probability_error_cheetah) + (BG_prior*probability_error_grass);
end