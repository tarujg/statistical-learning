%% Clear workspace and close windows

clc
clear
close all

%% Load training dataset

    DCT = load('TrainingSamplesDCT_8.mat');
    Train_DCT_FG = DCT.TrainsampleDCT_FG;
    Train_DCT_BG = DCT.TrainsampleDCT_BG;

%% The class priors for FG and BG are computed
%  Section A

    % Assuming prior distribution is (# of samples of A)/total training samples
    TotalNumberOfTraining_samples = size(Train_DCT_FG,1) + size(Train_DCT_BG,1);
    FG_prior = size(Train_DCT_FG,1)/TotalNumberOfTraining_samples
    BG_prior = size(Train_DCT_BG,1)/TotalNumberOfTraining_samples

%% Training data used for computing and plotting index histograms.
%  P(x|cheetah) and P(x|grass) are the class conditionals for FG and BG.
%  The feature is the index of DCT component with second largest magnitude.
%  The feature is the index of the DCT component with 2nd greatest energy.
    % Feature computation for BG sections
        X_BG = zeros(size(Train_DCT_BG,1),1);
        for idx = 1:size(Train_DCT_BG,1)
            X_BG(idx) = Index2ndLargest(Train_DCT_BG(idx,:));
        end
    
    % Feature computation for FG sections
        X_FG = zeros(size(Train_DCT_FG,1),1);
        for idx = 1:size(Train_DCT_FG,1)
            X_FG(idx) = Index2ndLargest(Train_DCT_FG(idx,:));
        end
    
    % Histogram plotting for both BG and FG
    % The bins for the histograms
        edges = 1:65;
        fontSize = 10;
        
        figure(1)
        
        h_BG = histogram(X_BG,'BinEdges',edges,'normalization', 'pdf','DisplayName','BG');
        BG_CCD = histcounts(X_BG,'BinEdges',edges, 'Normalization', 'probability');
        
        title('Class Conditional Distributions for Grass', 'FontSize', 1.5*fontSize);
        xlabel('Index of the DCT component with 2nd greatest energy', 'FontSize', fontSize);
        ylabel('P(x|grass)', 'FontSize', fontSize);
        legend('Grass')
        
        figure(2)
        
        h_FG = histogram(X_FG,'BinEdges',edges,'normalization', 'pdf','DisplayName','FG');
        FG_CCD = histcounts(X_FG,'BinEdges',edges, 'Normalization', 'probability');
        
        title('Class Conditional Distributions for Cheetah', 'FontSize', 1.5*fontSize);
        xlabel('Index of the DCT component with 2nd greatest energy', 'FontSize', fontSize);
        ylabel('P(x|cheetah)', 'FontSize', fontSize);
        legend('Cheetah')
        
%% For each block in the image cheetah.bmp, compute the feature X and state variable Y.
%  For Y use the minimum probability of error rule based on the distributions obtained above.
%  Store the state in an array A and then convert to binary image using imagesc and colormap(gray(255))
    
    block_size = 8;
    
    % Read the ZigZag pattern and convert to array and index from 1
        ZigZagPattern = table2array(readtable('Zig-Zag Pattern.txt'))+1;
    
    % Read image and convert to double
        image = im2double(imread('cheetah.bmp'));

    % Reading image dimensions
        [height,width] = size(image);
        [h_8,w_8] = deal(8*(ceil(height/8)+1),8*(ceil(width/8)+1));
        
    % Zeropad the image and convert to 8x8
        zeropad = zeros(h_8,w_8);
        zeropad(1:height,1:width) = image;
    
    % Create a blank array X
        X = zeros(height,width);
        dctZigZag = zeros(1,block_size*block_size);
    
    % Iterating through the image in 8x8 blocks through a sliding window
    for h = 1:height
        for w = 1:width
            
            dctBlock = dct2(zeropad(h:h+block_size-1,w:w+block_size-1));
            
            % Rearranging the DCT Components in ZigZag Patterned Vector
            for i = 1:block_size
                for j = 1:block_size
                    dctZigZag(ZigZagPattern(i,j)) = dctBlock(i,j);
                end
            end
            
            % Computing the feature for the block
            X(h,w) = Index2ndLargest(dctZigZag);

        end
    end
    
    % Computing the Posteriori distributions for generating predictions
    BG_posteriori = BG_CCD(X)*BG_prior;
    FG_posteriori = FG_CCD(X)*FG_prior;
    A = FG_posteriori > BG_posteriori;
    
    figure(3)
    imagesc(A);
    colormap(gray(255));
    imwrite(A, 'result.bmp');
    title('Predicted Segmentation based on Bayesian Decision Theory', 'FontSize', 1.5*fontSize);
        
    
%% Compare the ground truth in image cheetah_mask.bmp and compute the probability of error

    % Read the ground truth image
        ground_truth = im2double(imread('cheetah_mask.bmp'));
    
    % Array A has the mask and we compare where it differs from theground truth
        error_probability = mean(xor(A, ground_truth),"all")

%% UTILITY FUNCTIONS
    % 1. Index2ndLargest
    % Find Index of the second coefficient with second largest magnitude
    % Much faster than sorting and finding value at second position

    function [ind2] = Index2ndLargest(FeatureVector) % i is the x-largest value
        absFeatureVector = abs(FeatureVector);
        [~,ind1] = max(absFeatureVector);
        absFeatureVector(ind1) = -Inf;
        [~,ind2] = max(absFeatureVector);
    end