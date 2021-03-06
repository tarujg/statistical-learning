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
    
%% Pre-processing for each dataset

    % Datasets
    datasets = load('TrainingSamplesDCT_subsets_8.mat');
    % Load different Alphas
    load('Alpha.mat');        
    
    Bayes_error = zeros(size(alpha,2),1);
    MAP_error = zeros(size(alpha,2),1);
    MLE_error = zeros(size(alpha,2),1);
    
    Bayesian = zeros(height,width);
    MAP = zeros(height,width);
    MLE = zeros(height,width);
    
%% Running experiments for a particular subset and prior strategy
    for idx = 1:4
        switch idx
            case 1
                Train_FG = datasets.D1_FG;Train_BG = datasets.D1_BG;
            case 2
                Train_FG = datasets.D2_FG;Train_BG = datasets.D2_BG;
            case 3
                Train_FG = datasets.D3_FG;Train_BG = datasets.D3_BG;
            case 4
                Train_FG = datasets.D4_FG;Train_BG = datasets.D4_BG;
        end
        
        N_BG = size(Train_BG,1);
        N_FG = size(Train_FG,1);
    
        % Based on dataset size
        prior_BG = N_BG/(N_BG+N_FG);
        prior_FG = N_FG/(N_BG+N_FG);
    
        for strat_idx =1:2
            strategy = load(strcat('Prior_',num2str(strat_idx),'.mat'));
            for alpha_idx = 1:size(alpha,2)
    
                % Compute Hyperparameters for Prior based on different strategies
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

                subplot(1,3,1)
                imagesc(Bayesian);
                colormap(gray(255));
                imwrite(Bayesian(1:height,1:width), strcat('./results/',num2str(idx),'/bayes_',num2str(strat_idx),'_',num2str(alpha_idx),'.bmp'));
                title('Bayesian')

                subplot(1,3,2)
                imagesc(MLE);
                colormap(gray(255));
                imwrite(MLE(1:height,1:width), strcat('./results/',num2str(idx),'/mle_',num2str(strat_idx),'_',num2str(alpha_idx),'.bmp'));
                title('MLE')

                subplot(1,3,3)
                imagesc(MAP);
                colormap(gray(255));
                imwrite(MAP(1:height,1:width), strcat('./results/',num2str(idx),'/map_',num2str(strat_idx),'_',num2str(alpha_idx),'.bmp'));
                title('MAP')

                % Compare the ground truth in image cheetah_mask.bmp and compute the probability of error
                % Read the ground truth image
                Bayes_error(alpha_idx) = error_computation(ground_truth,Bayesian,prior_FG,prior_BG);
                MLE_error(alpha_idx) = error_computation(ground_truth,MLE,prior_FG,prior_BG);
                MAP_error(alpha_idx) = error_computation(ground_truth,MAP,prior_FG,prior_BG);

            end 
            
            x = 1:size(alpha,2);

            figure(strat_idx+2*(idx-1));
            semilogx(alpha(x), Bayes_error(x), '--r'), hold on
            semilogx(alpha(x), MAP_error(x), '-xg'), hold on
            semilogx(alpha(x), MLE_error(x), '-ob')
            title(strcat('Dataset ',num2str(idx),' with Strategy ',num2str(strat_idx)))
            xlabel('\alpha')
            ylabel('Probability of Error')
            legend('Bayesian','MAP','ML')
            saveas(gcf,strcat('./results/Strat_',num2str(strat_idx),'_Data_',num2str(idx),'.pdf'));
        end
    end
    
   
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