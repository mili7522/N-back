clc
clear all

% Read data
data = dlmread('../Data/100307.tsv', '\t', 1, 1)';

% Setup filter
fcuthigh = 0.01; % Hz 
sampling_rate = 1.3; % samples per seconds
Wn = [fcuthigh] / ( 1/2 * sampling_rate );
order = 3;
[b, a] = butter( order, Wn, 'high');

% Filter data (a (time x ROI) matrix)
data_zscore = zscore( data );
data_detrend = detrend( data_zscore );
data_filter = filtfilt( b, a, data_detrend );
data_meanremoval = data_filter - mean(data_filter, 2);

% Auto-correlation
d = data_zscore(:,1);  % Region 1
N = length(d);
autocorr_zscore = autocorr( d, N-1 );

d = data_detrend(:,1);
autocorr_detrend = autocorr( d, N-1 );

d = data_filter(:,1);
autocorr_filter = autocorr( d, N-1 );

d = data_meanremoval(:,1);
autocorr_meanremoval = autocorr( d, N-1 );

acl_zscore = zeros(333,1);
acl_detrend = zeros(333,1);
acl_filter = zeros(333,1);
acl_meanremoval = zeros(333,1);
% d is an Nx1 vector
for i = 1:333
    d = data_zscore(:,i);
    acl_zscore(i) = ceil( 2 * sum( autocorr( d, N-1 ).^2 ) ) - 2;
    
    d = data_detrend(:,i);
    acl_detrend(i) = ceil( 2 * sum( autocorr( d, N-1 ).^2 ) ) - 2;
    
    d = data_filter(:,i);
    acl_filter(i) = ceil( 2 * sum( autocorr( d, N-1 ).^2 ) ) - 2;
    
    d = data_meanremoval(:,i);
    acl_meanremoval(i) = ceil( 2 * sum( autocorr( d, N-1 ).^2 ) ) - 2;
end

acl_zscore_te = zeros(333,333);
acl_detrend_te = zeros(333,333);
acl_filter_te = zeros(333,333);
acl_meanremoval_te = zeros(333,333);
for i = 1:333
    for j = 1:333
        d1 = data_zscore(:,i);
        d2 = data_zscore(:,j);
        
        acl_zscore_te(i,j) = ceil( 2 * sum( autocorr( d1, N-1 ) .* autocorr( d2, N-1 ) ) ) - 2;

        d1 = data_detrend(:,i);
        d2 = data_detrend(:,j);
        acl_detrend_te(i,j) = ceil( 2 * sum( autocorr( d1, N-1 ) .* autocorr( d2, N-1 ) ) ) - 2;

        d1 = data_filter(:,i);
        d2 = data_filter(:,j);
        acl_filter_te(i,j) = ceil( 2 * sum( autocorr( d1, N-1 ) .* autocorr( d2, N-1 ) ) ) - 2;

        d1 = data_meanremoval(:,i);
        d2 = data_meanremoval(:,j);
        acl_meanremoval_te(i,j) = ceil( 2 * sum( autocorr( d1, N-1 ) .* autocorr( d2, N-1 ) ) ) - 2;
    end
end


save('../Preprocessing_steps_100307.mat')
