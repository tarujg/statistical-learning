%% Clear workspace and close windows

    clc
    clear
    close all
    
%% Feature Computation for 8x8 block in the image cheetah.bmp, compute the feature X.

    block_size = 8;
    ZigZagPattern = table2array(readtable('Zig-Zag Pattern.txt'))+1;
    
    image = im2double(imread('cheetah.bmp'));
    ground_truth = im2double(imread('cheetah_mask.bmp'));
    
    % Reading image dimensions and zeropadding
        [height,width] = size(image);
        [h_8,w_8] = deal(8*(ceil(height/8)+1),8*(ceil(width/8)+1));
        zeropad = zeros(h_8,w_8);
        zeropad(1:height,1:width) = image;
        dctZigZag = zeros(height,width,block_size*block_size);
    
    % Iterating through the image in 8x8 blocks through a sliding window
    for h = 1:height
        for w = 1:width
            
            dctBlock = dct2(zeropad(h:h+block_size-1,w:w+block_size-1));
            % Rearranging the DCT Components in ZigZag Patterned Vector
            for i = 1:block_size
                for j = 1:block_size
                    dctZigZag(h,w,ZigZagPattern(i,j)) = dctBlock(i,j);
                end
            end
        end
    end
    
%% Pre-processing for each dataset

    % Datasets i
    datasets = load('TrainingSamplesDCT_subsets_8.mat');
    Train_FG = datasets.D1_FG;
    Train_BG = datasets.D1_BG;
    
    N_BG = size(Train_BG,1);
    N_FG = size(Train_FG,1);
    
    % Based on dataset size
    prior_BG = N_BG/(N_BG+N_FG);
    prior_FG = N_FG/(N_BG+N_FG);
    
    strategy = load('Prior_1.mat');

    % Load different Alphas
    load('Alpha.mat');

    Bayes_error = zeros(size(alpha,2),1);
    MLE_error = zeros(size(alpha,2),1);
    MAP_error = zeros(size(alpha,2),1);
    
%% Running experiments for a particular subset and prior strategy

    for alpha_idx = 1:size(alpha,2)
        % Compute Hyperparameters for Prior based on different strategies
        
        Bayesian = zeros(height,width);
        MAP = zeros(height,width);
        MLE = zeros(height,width);
        
        W0 = strategy.W0;
        mu0_FG = strategy.mu0_FG;
        mu0_BG = strategy.mu0_BG;
        cov_00 = diag(alpha(alpha_idx)*W0);

        % Sample mean and Covariance
        sample_mu_BG = Train_BG'*(ones(N_BG,1))/N_BG;
        sample_mu_FG = Train_FG'*(ones(N_FG,1))/N_FG;
        sample_cov_BG = (Train_BG'*Train_BG)/N_BG - sample_mu_BG*sample_mu_BG';
        sample_cov_FG = (Train_FG'*Train_FG)/N_FG - sample_mu_FG*sample_mu_FG';
        sample_precision_BG = inv(sample_cov_BG);
        sample_precision_FG = inv(sample_cov_FG);
        
        sample_cov_BG_norm = sample_cov_BG/N_BG;
        sample_cov_FG_norm = sample_cov_FG/N_FG; 

        % Posterior mean and covariance
        posterior_mu_BG = cov_00*inv(cov_00+sample_cov_BG_norm)*sample_mu_BG + sample_cov_BG_norm*inv(cov_00+sample_cov_BG_norm)*mu0_BG';
        posterior_mu_FG = cov_00*inv(cov_00+sample_cov_FG_norm)*sample_mu_FG + sample_cov_FG_norm*inv(cov_00+sample_cov_FG_norm)*mu0_FG';
        posterior_cov_BG = cov_00*inv(cov_00+sample_cov_BG_norm)*sample_cov_BG_norm;
        posterior_cov_FG = cov_00*inv(cov_00+sample_cov_FG_norm)*sample_cov_FG_norm;

        % Predictive mean and covariance
        predictive_mu_BG = posterior_mu_BG;
        predictive_mu_FG = posterior_mu_FG;
        predictive_cov_BG = posterior_cov_BG + sample_cov_BG;
        predictive_cov_FG = posterior_cov_FG + sample_cov_FG;
        predictive_precision_BG = inv(predictive_cov_BG);
        predictive_precision_FG = inv(predictive_cov_FG);
        
        % Iterating through the image
        for h = 1:height
            for w = 1:width
                feature_vec = squeeze(dctZigZag(h,w,:));
                
                g_grass = decision_bound(feature_vec,predictive_cov_BG,predictive_mu_BG,prior_BG,predictive_precision_BG);
                g_cheetah = decision_bound(feature_vec,predictive_cov_FG,predictive_mu_FG,prior_FG,predictive_precision_FG);
                
                Bayesian(h,w) = g_grass > g_cheetah;
                
                g_grass = decision_bound(feature_vec,sample_cov_BG,sample_mu_BG,prior_BG,sample_precision_BG);
                g_cheetah = decision_bound(feature_vec,sample_cov_FG,sample_mu_FG,prior_FG,sample_precision_FG);
                
                MLE(h,w) = g_grass > g_cheetah;
              
                g_grass = decision_bound(feature_vec,sample_cov_BG,posterior_mu_BG,prior_BG,sample_precision_BG);
                g_cheetah = decision_bound(feature_vec,sample_cov_FG,posterior_mu_FG,prior_FG,sample_precision_FG);
                
                MAP(h,w) = g_grass > g_cheetah;
            end
        end
        
        % Displaying the output image for both feature sets
        A = Bayesian(1:height,1:width);
        figure(1)
        imagesc(A);
        colormap(gray(255));
        imwrite(Bayesian(1:height,1:width), strcat('./results/bayes_',num2str(alpha_idx),'_.bmp'));
        
        
        B = MLE(1:height,1:width);
        figure(1)
        imagesc(B);
        colormap(gray(255));
        imwrite(MLE(1:height,1:width), strcat('./results/mle_',num2str(alpha_idx),'_.bmp'));
        
        
        C = MAP(1:height,1:width);
        figure(1)
        imagesc(B);
        colormap(gray(255));
        imwrite(MAP(1:height,1:width), strcat('./results/map_',num2str(alpha_idx),'_.bmp'));
        
        % Compare the ground truth in image cheetah_mask.bmp and compute the probability of error
        % Read the ground truth image
        Bayes_error(alpha_idx) = error_computation(ground_truth,A,prior_FG,prior_BG);
        MLE_error(alpha_idx) = error_computation(ground_truth,B,prior_FG,prior_BG);
        MAP_error(alpha_idx) = error_computation(ground_truth,C,prior_FG,prior_BG);
        
    end    
    
    x = 1:size(alpha,2);

    figure;
    semilogx(alpha(x), Bayes_error(x), '--r'), hold on
    semilogx(alpha(x), MAP_error(x), '-xg'), hold on
    semilogx(alpha(x), MLE_error(x), '-ob')
    xlabel('\alpha')
    ylabel('Probability of Error')
    legend('Bayesian','MAP','ML')
   
%% UTILITY FUNCTIONS
    
    function [g_x] = decision_bound(x,Var,Mean,prior,precision) 
    
        w_i_0 = log(det(Var))-2*log(prior)+(Mean'*precision*Mean);
        w_i = -2*precision*Mean;
        g_x = x'*precision*x + w_i'*x + w_i_0;
    end
    
    function [probability_error] = error_computation(ground_truth,prediction,FG_prior,BG_prior)
        
        % Probability of error for Cheetah pixels misclassified as Grass
            probability_error_cheetah = sum(ground_truth & ~prediction,'all')/sum(ground_truth,'all');
        % Probability of error for Grass pixels misclassified as Cheetah
            probability_error_grass = sum(~ground_truth & prediction,'all')/sum(~ground_truth,'all');
        % Computation of probability of error
            probability_error = (FG_prior*probability_error_cheetah) + (BG_prior*probability_error_grass);
    end